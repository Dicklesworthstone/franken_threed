/**
 * @file tools/compat/mesh_adapter.test.mjs
 * Unit and regression test suite for dynamic Three.js Mesh adapter (f3d-05.6 / 19:26Z product wave).
 *
 * Uses real upstream Three.js classes to verify:
 * 1. Geometry, index, and clamped drawRange extraction from real BufferGeometry.
 * 2. Material color extraction from real default MeshBasicMaterial with depthTest=false, depthWrite=false, DoubleSide.
 * 3. Default Material methods (onBeforeCompile, customProgramCacheKey) are admitted; custom overrides are rejected.
 * 4. Visibility checks: mesh.visible=false and material.visible=false are rejected.
 * 5. Callback checks: mesh/material onBeforeRender and onAfterRender overrides are rejected (root 19:47Z).
 * 6. Layer checks: camera and mesh layer mismatch is rejected (root 19:47Z).
 * 7. Subclass checks: InstancedMesh, SkinnedMesh, and BatchedMesh are rejected (root 19:47Z).
 * 8. Model-View matrix multiplication from real Mesh and Camera transforms with explicit update boundaries.
 * 9. WebGL-to-WebGPU depth projection tracking.
 * 10. Dynamic mutation tracking: in-place mutations of geometry, material, world matrix,
 *     and camera change the outgoing snapshot without mutating prior snapshots.
 * 11. Indexed geometry ordering, vertex expansion, and out-of-bounds index rejection.
 * 12. Empty indexed drawRange passes empty positions (preventing unindexed draw fallback).
 * 13. Incomplete tail indices dropped in triangle expansion per native triangle-list semantics.
 * 14. Admission of DoubleSide only; explicit rejection of FrontSide and BackSide.
 * 15. Comprehensive negative rejections: interleaved/normalized attributes, vertexColors, colorWrite=false,
 *     clippingPlanes, alphaTest, alphaHash, custom blending, shader hooks, transparency, textures, morph targets, wireframe.
 * 16. Wasm packet preparation with exact 8 typed arguments and truthful retained fallback route.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.module.js';

import {
  canAdmitMesh,
  extractMeshRenderData,
  prepareMeshPacket,
  prepareMeshDepthPacket,
  prepareCanvasMeshPacket,
  prepareCanvasMeshDepthPacket,
  prepareMeshBatchPacket,
  renderMesh,
  renderMeshBatch,
  renderScene,
  renderRetainedFallback,
  multiplyMatrices4x4,
  expandIndexedPositions,
  ADMISSION_REJECTION,
  createAdmissionError,
  COORDINATE_SYSTEM,
  THREE_SIDE,
  CULL_MODE_WIRE,
  FRONT_FACE_WIRE,
  DEPTH_WIRE_COMPARE,
  THREE_DEPTH_FUNC_TO_WIRE_COMPARE,
  computeAffineDeterminant,
} from './mesh_adapter.mjs';

function createBasicTriangleMesh(materialProps = {}, geomProps = {}) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    0.0,  0.5, 0.0,
   -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide, // Mandatory per 13043 / 13062 point 1
    ...materialProps,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.updateMatrixWorld();
  return mesh;
}

function createDepthTriangleMesh(materialProps = {}, geomProps = {}) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    0.0,  0.5, 0.0,
   -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    side: THREE.DoubleSide, // Mandatory DoubleSide
    ...materialProps,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.updateMatrixWorld();
  return mesh;
}

function createBasicCamera(coordSystem = undefined) {
  const camera = new THREE.PerspectiveCamera(60, 1.0, 0.1, 100.0);
  if (coordSystem !== undefined) {
    camera.coordinateSystem = coordSystem;
  }
  camera.position.set(0, 0, 5);
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  return camera;
}

test('renderScene preserves layer-filtered group ordering and culls invisible materials', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();
  const group = new THREE.Group();
  group.layers.set(1);
  group.renderOrder = 10;
  const first = createBasicTriangleMesh();
  const second = createBasicTriangleMesh({ color: 0x0000ff });
  group.add(first);
  scene.add(group, second);
  const hiddenMaterial = createBasicTriangleMesh({ wireframe: true, visible: false });
  scene.add(hiddenMaterial);
  let submissions = 0;
  let vertexCounts;
  const host = { executePacket() { submissions++; } };
  const wasm = { f3d_build_mesh_batch_packet(_positions, counts) {
    vertexCounts = Array.from(counts);
    return new Uint8Array();
  } };
  const result = await renderScene(host, scene, camera, null, wasm, { sourceBackend: 'webgpu' });
  // The group is outside camera layers, but its child is visible; only the child's id breaks the z tie.
  assert.deepEqual(result.admitted, [first.uuid, second.uuid]);
  assert.deepEqual(result.refused, []);
  assert.deepEqual(vertexCounts, [3, 3]);
  assert.equal(submissions, 1);
});

test('renderScene refuses scene hooks before building or submitting a partial result', async () => {
  const camera = createBasicCamera();
  const scene = new THREE.Scene();
  scene.add(createBasicTriangleMesh());
  const fail = () => assert.fail('Refused scene must not invoke hooks, build packets, or submit');
  const host = { executePacket: fail };
  const wasm = { f3d_build_mesh_batch_packet: fail };
  for (const hook of ['onBeforeRender', 'onAfterRender']) {
    scene[hook] = fail;
    const result = await renderScene(host, scene, camera, null, wasm);
    assert.deepEqual(result.admitted, []);
    assert.equal(result.refused[0].code, 'UNSUPPORTED_CALLBACK');
    scene[hook] = THREE.Object3D.prototype[hook];
  }
});

test('Positive: Real default MeshBasicMaterial inherits prototype methods and is admitted without hook deletion', () => {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, depthTest: false, depthWrite: false });
  const mesh = new THREE.Mesh(geom, mat);
  const camera = createBasicCamera();

  // Verify prototype methods are genuinely present and inherited on the real instance
  assert.equal(typeof mat.onBeforeCompile, 'function');
  assert.equal(typeof mat.customProgramCacheKey, 'function');
  assert.equal(mat.onBeforeCompile, THREE.Material.prototype.onBeforeCompile);
  assert.equal(mat.customProgramCacheKey, THREE.Material.prototype.customProgramCacheKey);

  // Must be admitted!
  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, true, 'Default unchanged MeshBasicMaterial must be admitted');
});

test('Negative: material.visible === false is rejected with MATERIAL_NOT_VISIBLE (root 19:47Z point 1)', () => {
  const mesh = createBasicTriangleMesh({ visible: false });
  const camera = createBasicCamera();

  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, false);
  assert.equal(admission.reason, ADMISSION_REJECTION.MATERIAL_NOT_VISIBLE);
});

test('Negative: mesh.onBeforeRender and onAfterRender callbacks are rejected (root 19:47Z point 2)', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  // mesh.onBeforeRender override
  mesh.onBeforeRender = () => { mesh.material.color.setRGB(0, 1, 0); };
  const beforeAdmission = canAdmitMesh(mesh, camera);
  assert.equal(beforeAdmission.admitted, false);
  assert.equal(beforeAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_CALLBACK);

  // Clean mesh, add onAfterRender override
  const mesh2 = createBasicTriangleMesh();
  mesh2.onAfterRender = () => {};
  const afterAdmission = canAdmitMesh(mesh2, camera);
  assert.equal(afterAdmission.admitted, false);
  assert.equal(afterAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_CALLBACK);

  // material.onBeforeRender override
  const mesh3 = createBasicTriangleMesh();
  mesh3.material.onBeforeRender = () => {};
  const matBeforeAdmission = canAdmitMesh(mesh3, camera);
  assert.equal(matBeforeAdmission.admitted, false);
  assert.equal(matBeforeAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_CALLBACK);
});

test('Negative: Layer mismatch between camera and mesh is rejected (root 19:47Z)', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  // Mesh put on layer 1, camera remains on default layer 0
  mesh.layers.set(1);
  assert.equal(camera.layers.test(mesh.layers), false);

  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, false);
  assert.equal(admission.reason, ADMISSION_REJECTION.LAYER_MISMATCH);
});

test('Negative: InstancedMesh, SkinnedMesh, and BatchedMesh subclasses are rejected (root 19:47Z)', () => {
  const camera = createBasicCamera();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, depthTest: false, depthWrite: false });

  // 1. InstancedMesh
  const instMesh = new THREE.InstancedMesh(geom, mat, 2);
  const instAdmission = canAdmitMesh(instMesh, camera);
  assert.equal(instAdmission.admitted, false);
  assert.equal(instAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_MESH_SUBCLASS);

  // 2. SkinnedMesh
  const skinnedMesh = new THREE.SkinnedMesh(geom, mat);
  const skinnedAdmission = canAdmitMesh(skinnedMesh, camera);
  assert.equal(skinnedAdmission.admitted, false);
  assert.equal(skinnedAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_MESH_SUBCLASS);

  // 3. BatchedMesh
  const batchedMesh = new THREE.BatchedMesh(2, 10, 10, mat);
  const batchedAdmission = canAdmitMesh(batchedMesh, camera);
  assert.equal(batchedAdmission.admitted, false);
  assert.equal(batchedAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_MESH_SUBCLASS);
});

test('Negative: Custom onBeforeCompile and customProgramCacheKey overrides are rejected', () => {
  const camera = createBasicCamera();

  // Custom onBeforeCompile instance override
  const hookMesh = createBasicTriangleMesh();
  hookMesh.material.onBeforeCompile = (shader) => { shader.fragmentShader += ''; };
  const hookAdmission = canAdmitMesh(hookMesh, camera);
  assert.equal(hookAdmission.admitted, false);
  assert.equal(hookAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE);

  // Custom customProgramCacheKey instance override
  const cacheMesh = createBasicTriangleMesh();
  cacheMesh.material.customProgramCacheKey = () => 'custom_cache_key';
  const cacheAdmission = canAdmitMesh(cacheMesh, camera);
  assert.equal(cacheAdmission.admitted, false);
  assert.equal(cacheAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_MATERIAL_FEATURE);
});

test('Negative: mesh.visible === false is rejected with NOT_VISIBLE (Chartreuse 13058)', () => {
  const mesh = createBasicTriangleMesh();
  mesh.visible = false;
  const camera = createBasicCamera();

  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, false);
  assert.equal(admission.reason, ADMISSION_REJECTION.NOT_VISIBLE);
});

test('Positive: Extracts unindexed Three.js Mesh with correct geometry, color, and matrices', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, true, 'Real unindexed Mesh must be admitted');

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);

  assert.equal(snapshot.vertexCount, 3);
  assert.equal(snapshot.triangleCount, 1);
  assert.equal(snapshot.isIndexed, false);
  assert.equal(snapshot.indices.length, 0);
  assert.equal(snapshot.positions.length, 9);
  assert.deepEqual(Array.from(snapshot.positions), [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0]);

  // Color: Red [1, 0, 0, 1]
  assert.equal(snapshot.color.length, 4);
  assert.equal(snapshot.color[0], 1.0);
  assert.equal(snapshot.color[1], 0.0);
  assert.equal(snapshot.color[2], 0.0);
  assert.equal(snapshot.color[3], 1.0);

  // Model-View Matrix: 16 f64 elements
  assert.equal(snapshot.modelView.length, 16);
  assert.ok(snapshot.modelView instanceof Float64Array);
  assert.equal(snapshot.modelView[12], 0);
  assert.equal(snapshot.modelView[13], 0);
  assert.equal(snapshot.modelView[14], -5);

  // Projection matrix & WebGL depth flag
  assert.equal(snapshot.projection.length, 16);
  assert.ok(snapshot.projection instanceof Float64Array);
  assert.equal(snapshot.webglDepth, true, 'Default Three.js camera must have webglDepth === true');
});

test('Positive: Extracts indexed Three.js Mesh with strict ordering and expansion', () => {
  const geometry = new THREE.BufferGeometry();
  // Quad vertices (4 vertices)
  const vertices = new Float32Array([
    -1, -1, 0,
     1, -1, 0,
     1,  1, 0,
    -1,  1, 0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  // Two triangles: (0, 1, 2) and (2, 3, 0)
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  const mesh = createBasicTriangleMesh({ color: 0x0000ff });
  mesh.geometry = geometry;
  mesh.updateMatrixWorld();
  const camera = createBasicCamera();

  const snapshot = extractMeshRenderData(mesh, camera, 128, 128);

  assert.equal(snapshot.isIndexed, true);
  assert.equal(snapshot.triangleCount, 2);
  assert.equal(snapshot.indices.length, 6);
  assert.deepEqual(Array.from(snapshot.indices), [0, 1, 2, 2, 3, 0]);

  // Expanded unindexed positions
  assert.equal(snapshot.expandedPositions.length, 18);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(0, 3)), [-1, -1, 0]);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(3, 6)), [1, -1, 0]);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(6, 9)), [1, 1, 0]);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(9, 12)), [1, 1, 0]);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(12, 15)), [-1, 1, 0]);
  assert.deepEqual(Array.from(snapshot.expandedPositions.slice(15, 18)), [-1, -1, 0]);
});

test('Indexed snapshots preserve raw element indices when the attribute is normalized', () => {
  const camera = createBasicCamera();
  for (const ArrayType of [Uint16Array, Uint32Array]) {
    const mesh = createBasicTriangleMesh();
    const index = new THREE.BufferAttribute(new ArrayType([2, 2, 2, 0, 1, 2]), 1, true);
    mesh.geometry.setIndex(index);
    mesh.geometry.setDrawRange(3, 3);
    assert.ok(index.getX(4) < 1, 'the normalized accessor differs from the element buffer');

    const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
    assert.deepEqual(Array.from(snapshot.indices), [0, 1, 2]);
    assert.deepEqual(snapshot.expandedPositions, mesh.geometry.attributes.position.array);
    assert.equal(index.normalized, true, 'extraction must not mutate the source attribute');

    index.array[5] = 3;
    assert.throws(() => extractMeshRenderData(mesh, camera, 64, 64), /Index references vertex out of bounds/);
    assert.deepEqual(Array.from(snapshot.indices), [0, 1, 2], 'prior snapshots retain their indices');
  }
});

test('Positive: Empty indexed drawRange passes empty positions (root review invariant)', () => {
  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2]), 1));
  // Empty drawRange on indexed geometry
  geometry.setDrawRange(0, 0);

  const mesh = createBasicTriangleMesh();
  mesh.geometry = geometry;
  mesh.updateMatrixWorld();
  const camera = createBasicCamera();

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snapshot.positions.length, 0, 'Empty indexed range must produce empty positions');
  assert.equal(snapshot.indices.length, 0, 'Empty indexed range must produce empty indices');
  assert.equal(snapshot.expandedPositions.length, 0);
  assert.equal(snapshot.triangleCount, 0);
  assert.equal(snapshot.vertexCount, 0);
});

test('Positive: Incomplete tail indices dropped in triangle expansion per native triangle-list semantics', () => {
  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  // 5 indices: first 3 make 1 triangle; last 2 (indices 2, 3) form an incomplete tail and must be dropped
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 2, 3]), 1));

  const mesh = createBasicTriangleMesh();
  mesh.geometry = geometry;
  mesh.updateMatrixWorld();
  const camera = createBasicCamera();

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snapshot.triangleCount, 1);
  assert.equal(snapshot.indices.length, 5); // raw indices preserved
  assert.equal(snapshot.expandedPositions.length, 9); // exactly 1 triangle (3 vertices * 3 coords)
});

test('Side admission: FrontSide (0), BackSide (1), and DoubleSide (2) are admitted; invalid sides rejected', () => {
  const camera = createBasicCamera();

  const doubleMesh = createBasicTriangleMesh({ side: THREE.DoubleSide });
  assert.equal(canAdmitMesh(doubleMesh, camera).admitted, true);
  const doubleSnap = extractMeshRenderData(doubleMesh, camera, 64, 64);
  assert.equal(doubleSnap.side, THREE_SIDE.DOUBLE_SIDE);
  assert.equal(doubleSnap.cullMode, CULL_MODE_WIRE.NONE);

  const frontMesh = createBasicTriangleMesh({ side: THREE.FrontSide });
  assert.equal(canAdmitMesh(frontMesh, camera).admitted, true);
  const frontSnap = extractMeshRenderData(frontMesh, camera, 64, 64);
  assert.equal(frontSnap.side, THREE_SIDE.FRONT_SIDE);
  assert.equal(frontSnap.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(frontSnap.frontFace, FRONT_FACE_WIRE.CCW);

  const backMesh = createBasicTriangleMesh({ side: THREE.BackSide });
  assert.equal(canAdmitMesh(backMesh, camera).admitted, true);
  const backSnap = extractMeshRenderData(backMesh, camera, 64, 64);
  assert.equal(backSnap.side, THREE_SIDE.BACK_SIDE);
  assert.equal(backSnap.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(backSnap.frontFace, FRONT_FACE_WIRE.CW);

  // Default side (undefined) defaults to FrontSide
  const defaultSideMesh = createBasicTriangleMesh();
  delete defaultSideMesh.material.side;
  assert.equal(canAdmitMesh(defaultSideMesh, camera).admitted, true);
  const defaultSnap = extractMeshRenderData(defaultSideMesh, camera, 64, 64);
  assert.equal(defaultSnap.side, THREE_SIDE.FRONT_SIDE);

  // Invalid sides: 99, -1, 'unsupported'
  const invalidMesh99 = createBasicTriangleMesh({ side: 99 });
  const admission99 = canAdmitMesh(invalidMesh99, camera);
  assert.equal(admission99.admitted, false);
  assert.equal(admission99.reason, ADMISSION_REJECTION.UNSUPPORTED_SIDE);
  assert.throws(
    () => extractMeshRenderData(invalidMesh99, camera, 64, 64),
    /UNSUPPORTED_SIDE/
  );

  const invalidMeshNeg = createBasicTriangleMesh({ side: -1 });
  assert.equal(canAdmitMesh(invalidMeshNeg, camera).admitted, false);

  const invalidMeshStr = createBasicTriangleMesh({ side: 'two-sided' });
  assert.equal(canAdmitMesh(invalidMeshStr, camera).admitted, false);
});

test('drawRange correctness: Clamps Infinity count with nonzero start and preserves empty draw (13062 point 2)', () => {
  const geometry = new THREE.BufferGeometry();
  // 6 vertices = 2 triangles
  const vertices = new Float32Array([
    0, 0, 0,  1, 0, 0,  0, 1, 0,
    1, 1, 0,  2, 1, 0,  1, 2, 0,
  ]);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  // Start at vertex 3, count Infinity: should extract remaining 3 vertices, NOT overrun
  geometry.setDrawRange(3, Infinity);

  const mesh = createBasicTriangleMesh();
  mesh.geometry = geometry;
  mesh.updateMatrixWorld();
  const camera = createBasicCamera();

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snapshot.vertexCount, 3);
  assert.equal(snapshot.positions.length, 9);
  assert.deepEqual(Array.from(snapshot.positions), [1, 1, 0, 2, 1, 0, 1, 2, 0]);

  // Empty draw: count = 0
  geometry.setDrawRange(0, 0);
  const emptySnap = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(emptySnap.vertexCount, 0);
  assert.equal(emptySnap.triangleCount, 0);
  assert.equal(emptySnap.positions.length, 0);

  // Invalid drawRange negative values must throw
  geometry.setDrawRange(-1, 3);
  assert.throws(
    () => extractMeshRenderData(mesh, camera, 64, 64),
    /Invalid drawRange: start and count must be non-negative integers/,
  );
});

test('Attribute validation: Rejects interleaved and normalized attributes (13062 point 3)', () => {
  const camera = createBasicCamera();
  const mesh = createBasicTriangleMesh();

  // Normalized attribute
  mesh.geometry.attributes.position.normalized = true;
  assert.equal(canAdmitMesh(mesh, camera).admitted, false);
  assert.equal(canAdmitMesh(mesh, camera).reason, ADMISSION_REJECTION.UNSUPPORTED_ATTRIBUTE);
  assert.throws(
    () => extractMeshRenderData(mesh, camera, 64, 64),
    /Interleaved or normalized vertex attributes are not supported/,
  );

  // Interleaved attribute
  mesh.geometry.attributes.position.normalized = false;
  mesh.geometry.attributes.position.isInterleavedBufferAttribute = true;
  assert.equal(canAdmitMesh(mesh, camera).admitted, false);
  assert.equal(canAdmitMesh(mesh, camera).reason, ADMISSION_REJECTION.UNSUPPORTED_ATTRIBUTE);
});

test('Index bounds check: Explicitly rejects out-of-range indices (13062 point 4)', () => {
  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]); // 3 vertices (indices 0, 1, 2)
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  // Index 5 is out of bounds!
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 5]), 1));

  const mesh = createBasicTriangleMesh();
  mesh.geometry = geometry;
  mesh.updateMatrixWorld();
  const camera = createBasicCamera();

  assert.throws(
    () => extractMeshRenderData(mesh, camera, 64, 64),
    /Index references vertex out of bounds: index 5 at position 2 exceeds vertex count 3/,
  );
});

test('Material features: Rejects advanced and unexercised material features (13062 point 5)', () => {
  const camera = createBasicCamera();

  // vertexColors
  const vcMesh = createBasicTriangleMesh({ vertexColors: true });
  assert.equal(canAdmitMesh(vcMesh, camera).admitted, false);

  // colorWrite = false (admitted in this slice)
  const cwMesh = createBasicTriangleMesh({ colorWrite: false });
  assert.equal(canAdmitMesh(cwMesh, camera).admitted, true);

  // clippingPlanes
  const clipMesh = createBasicTriangleMesh({ clippingPlanes: [new THREE.Plane()] });
  assert.equal(canAdmitMesh(clipMesh, camera).admitted, false);

  // alphaTest > 0
  const atMesh = createBasicTriangleMesh({ alphaTest: 0.5 });
  assert.equal(canAdmitMesh(atMesh, camera).admitted, false);

  // alphaHash = true
  const ahMesh = createBasicTriangleMesh({ alphaHash: true });
  assert.equal(canAdmitMesh(ahMesh, camera).admitted, false);

  // Custom blending != 1
  const blendMesh = createBasicTriangleMesh({ blending: THREE.AdditiveBlending });
  assert.equal(canAdmitMesh(blendMesh, camera).admitted, false);
});

test('Positive: Explicit matrix update boundary and mutation tracking (13062 point 6)', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  // Snapshot 1
  const snap1 = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snap1.color[0], 1.0); // Red
  assert.equal(snap1.positions[0], 0.0);
  assert.equal(snap1.modelView[12], 0.0);

  // In-place mutation with explicit update boundary
  mesh.material.color.setHex(0x00ff00);
  mesh.geometry.attributes.position.setX(0, 0.75);
  mesh.position.set(3, 0, 0);
  mesh.updateMatrixWorld(true);

  camera.position.set(0, 2, 10);
  camera.updateMatrixWorld(true);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();

  // Snapshot 2 reflects updated source state
  const snap2 = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snap2.color[0], 0.0);
  assert.equal(snap2.color[1], 1.0); // Green
  assert.equal(snap2.positions[0], 0.75);
  assert.equal(snap2.modelView[12], 3.0);
  assert.equal(snap2.modelView[13], -2.0);
  assert.equal(snap2.modelView[14], -10.0);

  // Prior snapshot is completely isolated
  assert.equal(snap1.color[0], 1.0);
  assert.equal(snap1.positions[0], 0.0);
  assert.equal(snap1.modelView[12], 0.0);
});

test('Positive: WebGPUCoordinateSystem camera sets webglDepth = false', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera(COORDINATE_SYSTEM.WEBGPU);

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snapshot.webglDepth, false, 'WebGPU camera must have webglDepth === false');
});

test('Positive: prepareMeshPacket invokes Wasm export with exact 8 typed arguments', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_packet: (positions, indices, mv, proj, color, w, h, webglDepth) => {
      capturedArgs = { positions, indices, mv, proj, color, w, h, webglDepth };
      return new Uint8Array([0x46, 0x33, 0x44, 0x31]); // 'F3D1'
    },
  };

  const { packetBytes, snapshot } = prepareMeshPacket(mesh, camera, 320, 240, mockWasm);

  assert.ok(capturedArgs, 'Wasm export must have been called');
  assert.equal(capturedArgs.positions.length, 9);
  assert.equal(capturedArgs.indices.length, 0);
  assert.equal(capturedArgs.mv.length, 16);
  assert.equal(capturedArgs.proj.length, 16);
  assert.equal(capturedArgs.color.length, 4);
  assert.equal(capturedArgs.w, 320);
  assert.equal(capturedArgs.h, 240);
  assert.equal(capturedArgs.webglDepth, true);
  assert.deepEqual(Array.from(packetBytes), [0x46, 0x33, 0x44, 0x31]);
  assert.equal(snapshot.vertexCount, 3);
});

test('Positive: renderRetainedFallback preserves JS ownership and invokes upstream renderer', () => {
  let renderCalled = false;
  let passedScene = null;
  let passedCamera = null;

  const mockRenderer = {
    render: (scene, cam) => {
      renderCalled = true;
      passedScene = scene;
      passedCamera = cam;
    },
  };

  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const report = renderRetainedFallback(mockRenderer, scene, camera);

  assert.equal(renderCalled, true);
  assert.equal(passedScene, scene);
  assert.equal(passedCamera, camera);
  assert.equal(report.implementationOwner, 'retained-js');
  assert.equal(report.rendered, true);
});

test('Negative: renderMesh rejects canvasContext when wasmModule lacks f3d_build_canvas_mesh_packet (no silent offscreen-as-visible)', async () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  let executeCalled = false;
  const mockBridgeHost = {
    executePacket: async () => {
      executeCalled = true;
      return { success: true };
    },
  };

  // Mock Wasm that ONLY exposes offscreen packet builder
  let offscreenCalled = false;
  const mockOffscreenWasm = {
    f3d_build_mesh_packet: () => {
      offscreenCalled = true;
      return new Uint8Array([1, 2, 3]);
    },
  };

  const mockCanvasContext = {
    canvas: { width: 128, height: 128 },
  };

  // Must reject when canvasContext is provided without canvas Wasm export
  await assert.rejects(
    async () => {
      await renderMesh(mockBridgeHost, mesh, camera, mockCanvasContext, mockOffscreenWasm);
    },
    {
      name: 'Error',
      message: /canvasContext provided for visible canvas rendering, but wasmModule does not export f3d_build_canvas_mesh_packet/,
    }
  );

  // Invariant: Bridge must NEVER have executed and offscreen packet must NEVER have been emitted
  assert.equal(executeCalled, false, 'Bridge executePacket must not be called on canvas rejection');
  assert.equal(offscreenCalled, false, 'Silent offscreen packet must not be generated for canvas target');
});

test('Negative: prepareCanvasMeshPacket and prepareMeshPacket(target: canvas) reject wasmModule lacking canvas export', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  const mockOffscreenWasm = {
    f3d_build_mesh_packet: () => new Uint8Array([1, 2, 3]),
  };

  // 1. prepareCanvasMeshPacket
  assert.throws(
    () => {
      prepareCanvasMeshPacket(mesh, camera, 64, 64, mockOffscreenWasm);
    },
    {
      name: 'Error',
      message: /wasmModule is missing f3d_build_canvas_mesh_packet export/,
    }
  );

  // 2. prepareMeshPacket with target: 'canvas'
  assert.throws(
    () => {
      prepareMeshPacket(mesh, camera, 64, 64, mockOffscreenWasm, { target: 'canvas' });
    },
    {
      name: 'Error',
      message: /wasmModule is missing f3d_build_canvas_mesh_packet export/,
    }
  );
});

test('Positive: renderMesh routes through f3d_build_canvas_mesh_packet when canvasContext is provided and export exists', async () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  let canvasExportCalled = false;
  let capturedArgs = null;
  const mockCanvasWasm = {
    f3d_build_mesh_packet: () => {
      throw new Error('Should not call offscreen builder');
    },
    f3d_build_canvas_mesh_packet: (positions, indices, mv, proj, color, w, h, webglDepth) => {
      canvasExportCalled = true;
      capturedArgs = { positions, indices, mv, proj, color, w, h, webglDepth };
      return new Uint8Array([0x43, 0x41, 0x4e, 0x56]); // 'CANV'
    },
  };

  let executedPacket = null;
  let executedContext = null;
  const mockBridgeHost = {
    executePacket: async (packet, ctx) => {
      executedPacket = packet;
      executedContext = ctx;
      return { renderedToCanvas: true };
    },
  };

  const mockCanvasContext = {
    canvas: { width: 256, height: 256 },
  };

  const { result, snapshot, target } = await renderMesh(
    mockBridgeHost,
    mesh,
    camera,
    mockCanvasContext,
    mockCanvasWasm
  );

  assert.equal(canvasExportCalled, true, 'Must call f3d_build_canvas_mesh_packet');
  assert.equal(capturedArgs.w, 256);
  assert.equal(capturedArgs.h, 256);
  assert.equal(target, 'canvas');
  assert.equal(executedContext, mockCanvasContext);
  assert.deepEqual(Array.from(executedPacket), [0x43, 0x41, 0x4e, 0x56]);
  assert.deepEqual(result, { renderedToCanvas: true });
});

test('Positive: renderMesh executes honest offscreen rendering when canvasContext is null or undefined', async () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  let offscreenExportCalled = false;
  const mockWasm = {
    f3d_build_mesh_packet: (positions, indices, mv, proj, color, w, h, webglDepth) => {
      offscreenExportCalled = true;
      return new Uint8Array([0x4f, 0x46, 0x46, 0x53]); // 'OFFS'
    },
  };

  let executedPacket = null;
  let executedContext = null;
  const mockBridgeHost = {
    executePacket: async (packet, ctx) => {
      executedPacket = packet;
      executedContext = ctx;
      return { readbackBufferId: 20 };
    },
  };

  // 1. canvasContext === null
  const resNull = await renderMesh(mockBridgeHost, mesh, camera, null, mockWasm, { width: 64, height: 64 });
  assert.equal(offscreenExportCalled, true);
  assert.equal(resNull.target, 'offscreen');
  assert.equal(executedContext, null);
  assert.deepEqual(Array.from(executedPacket), [0x4f, 0x46, 0x46, 0x53]);

  // 2. canvasContext === undefined
  offscreenExportCalled = false;
  const resUndef = await renderMesh(mockBridgeHost, mesh, camera, undefined, mockWasm, { width: 64, height: 64 });
  assert.equal(offscreenExportCalled, true);
  assert.equal(resUndef.target, 'offscreen');
  assert.equal(executedContext, null);
});

test('Positive: Real default MeshBasicMaterial with default depth settings is admitted and extracted', () => {
  const mesh = createDepthTriangleMesh();
  const camera = createBasicCamera();

  // Verify Three.js default material properties
  assert.equal(mesh.material.depthTest, true);
  assert.equal(mesh.material.depthWrite, true);
  assert.equal(mesh.material.depthFunc, THREE.LessEqualDepth);

  const admission = canAdmitMesh(mesh, camera);
  assert.equal(admission.admitted, true, 'Default MeshBasicMaterial with depthTest/depthWrite must be admitted');

  const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snapshot.depthTest, true);
  assert.equal(snapshot.depthWrite, true);
  assert.equal(snapshot.depthFunc, THREE.LessEqualDepth);
  assert.equal(snapshot.depthCompare, DEPTH_WIRE_COMPARE.LESS_EQUAL); // wire 4
});

test('Positive: All 8 Three.js depth functions map faithfully to wire compare codes 1..8', () => {
  const camera = createBasicCamera();
  const cases = [
    { name: 'NeverDepth', func: THREE.NeverDepth, expectedWire: DEPTH_WIRE_COMPARE.NEVER, wireCode: 1 },
    { name: 'AlwaysDepth', func: THREE.AlwaysDepth, expectedWire: DEPTH_WIRE_COMPARE.ALWAYS, wireCode: 8 },
    { name: 'LessDepth', func: THREE.LessDepth, expectedWire: DEPTH_WIRE_COMPARE.LESS, wireCode: 2 },
    { name: 'LessEqualDepth', func: THREE.LessEqualDepth, expectedWire: DEPTH_WIRE_COMPARE.LESS_EQUAL, wireCode: 4 },
    { name: 'EqualDepth', func: THREE.EqualDepth, expectedWire: DEPTH_WIRE_COMPARE.EQUAL, wireCode: 3 },
    { name: 'GreaterEqualDepth', func: THREE.GreaterEqualDepth, expectedWire: DEPTH_WIRE_COMPARE.GREATER_EQUAL, wireCode: 7 },
    { name: 'GreaterDepth', func: THREE.GreaterDepth, expectedWire: DEPTH_WIRE_COMPARE.GREATER, wireCode: 5 },
    { name: 'NotEqualDepth', func: THREE.NotEqualDepth, expectedWire: DEPTH_WIRE_COMPARE.NOT_EQUAL, wireCode: 6 },
  ];

  for (const c of cases) {
    const mesh = createDepthTriangleMesh({ depthFunc: c.func });
    const admission = canAdmitMesh(mesh, camera);
    assert.equal(admission.admitted, true, `${c.name} must be admitted`);

    const snapshot = extractMeshRenderData(mesh, camera, 64, 64);
    assert.equal(snapshot.depthFunc, c.func, `${c.name} depthFunc mismatch`);
    assert.equal(snapshot.depthCompare, c.expectedWire, `${c.name} wire compare mismatch`);
    assert.equal(snapshot.depthCompare, c.wireCode, `${c.name} wire code mismatch`);
    assert.equal(THREE_DEPTH_FUNC_TO_WIRE_COMPARE[c.func], c.expectedWire);
  }
});

test('Positive: depthTest=false forces depthCompare to ALWAYS (8) and resolves depthWrite via sourceBackend', () => {
  const camera = createBasicCamera();

  // 1. When options.sourceBackend === 'webgpu': WebGPU passes depthWrite directly (depthWriteEnabled = true)
  for (const func of [THREE.NeverDepth, THREE.LessDepth, THREE.GreaterDepth, THREE.EqualDepth]) {
    const mesh = createDepthTriangleMesh({ depthTest: false, depthWrite: true, depthFunc: func });
    assert.equal(canAdmitMesh(mesh, camera, { sourceBackend: 'webgpu' }).admitted, true);

    const snapshot = extractMeshRenderData(mesh, camera, 64, 64, { sourceBackend: 'webgpu' });
    assert.equal(snapshot.depthTest, false);
    assert.equal(snapshot.depthWrite, true, 'WebGPU backend must preserve depthWrite=true');
    assert.equal(snapshot.depthFunc, func);
    assert.equal(snapshot.depthCompare, DEPTH_WIRE_COMPARE.ALWAYS); // wire 8
  }

  // 2. When options.sourceBackend === 'webgl': WebGL hardware suppresses writes when depthTest is false
  for (const func of [THREE.NeverDepth, THREE.LessDepth, THREE.GreaterDepth, THREE.EqualDepth]) {
    const mesh = createDepthTriangleMesh({ depthTest: false, depthWrite: true, depthFunc: func });
    assert.equal(canAdmitMesh(mesh, camera, { sourceBackend: 'webgl' }).admitted, true);

    const snapshot = extractMeshRenderData(mesh, camera, 64, 64, { sourceBackend: 'webgl' });
    assert.equal(snapshot.depthTest, false);
    assert.equal(snapshot.depthWrite, false, 'WebGL backend must suppress depthWrite to false when depthTest is false');
    assert.equal(snapshot.depthFunc, func);
    assert.equal(snapshot.depthCompare, DEPTH_WIRE_COMPARE.ALWAYS); // wire 8
  }

  // 3. Negative: when depthTest=false and depthWrite=true without sourceBackend, must refuse execution
  const ambiguousMesh = createDepthTriangleMesh({ depthTest: false, depthWrite: true });
  const ambiguousAdm = canAdmitMesh(ambiguousMesh, camera);
  assert.equal(ambiguousAdm.admitted, false);
  assert.equal(ambiguousAdm.reason, ADMISSION_REJECTION.AMBIGUOUS_DEPTH_PAIR);
  assert.throws(
    () => extractMeshRenderData(ambiguousMesh, camera, 64, 64),
    /ambiguous across backends/,
  );

  // 4. For depthTest=false with depthWrite=false, no backend parameter needed
  const noDepthMesh = createDepthTriangleMesh({ depthTest: false, depthWrite: false });
  assert.equal(canAdmitMesh(noDepthMesh, camera).admitted, true);
  const noDepthSnap = extractMeshRenderData(noDepthMesh, camera, 64, 64);
  assert.equal(noDepthSnap.depthTest, false);
  assert.equal(noDepthSnap.depthWrite, false);
  assert.equal(noDepthSnap.depthCompare, DEPTH_WIRE_COMPARE.ALWAYS);
});

test('Positive: Dynamic depth mutations produce fresh isolated snapshots', () => {
  const mesh = createDepthTriangleMesh(); // Default: depthTest=true, depthWrite=true, depthFunc=LessEqual
  const camera = createBasicCamera();

  // Snapshot 1: Default
  const snap1 = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snap1.depthTest, true);
  assert.equal(snap1.depthWrite, true);
  assert.equal(snap1.depthFunc, THREE.LessEqualDepth);
  assert.equal(snap1.depthCompare, 4);

  // In-place mutation: change to GreaterDepth and depthWrite=false
  mesh.material.depthFunc = THREE.GreaterDepth;
  mesh.material.depthWrite = false;

  // Snapshot 2: Reflects mutation
  const snap2 = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snap2.depthTest, true);
  assert.equal(snap2.depthWrite, false);
  assert.equal(snap2.depthFunc, THREE.GreaterDepth);
  assert.equal(snap2.depthCompare, 5);

  // Invariant: Snapshot 1 is immutable and unaffected
  assert.equal(snap1.depthTest, true);
  assert.equal(snap1.depthWrite, true);
  assert.equal(snap1.depthFunc, THREE.LessEqualDepth);
  assert.equal(snap1.depthCompare, 4);

  // In-place mutation: disable depthTest
  mesh.material.depthTest = false;

  // Snapshot 3: depthTest=false -> depthCompare=ALWAYS (8)
  const snap3 = extractMeshRenderData(mesh, camera, 64, 64);
  assert.equal(snap3.depthTest, false);
  assert.equal(snap3.depthWrite, false);
  assert.equal(snap3.depthFunc, THREE.GreaterDepth);
  assert.equal(snap3.depthCompare, 8);

  // Invariant: Snapshots 1 and 2 remain unaffected
  assert.equal(snap1.depthCompare, 4);
  assert.equal(snap2.depthCompare, 5);
});

test('Negative: Unsupported stencil, polygonOffset, reversedDepth, and invalid depthFunc are rejected', () => {
  const camera = createBasicCamera();

  // 1. stencilWrite === true
  const stencilMesh = createDepthTriangleMesh({ stencilWrite: true });
  const stencilAdm = canAdmitMesh(stencilMesh, camera);
  assert.equal(stencilAdm.admitted, false);
  assert.equal(stencilAdm.reason, ADMISSION_REJECTION.UNSUPPORTED_STENCIL);

  // 2. polygonOffset === true
  const polyMesh = createDepthTriangleMesh({ polygonOffset: true });
  const polyAdm = canAdmitMesh(polyMesh, camera);
  assert.equal(polyAdm.admitted, false);
  assert.equal(polyAdm.reason, ADMISSION_REJECTION.UNSUPPORTED_POLYGON_OFFSET);

  // 3. camera.reversedDepth === true (via underlying _reversedDepth)
  const revCam1 = createBasicCamera();
  revCam1._reversedDepth = true;
  assert.equal(revCam1.reversedDepth, true);
  const normalMesh = createDepthTriangleMesh();
  const revAdm1 = canAdmitMesh(normalMesh, revCam1);
  assert.equal(revAdm1.admitted, false);
  assert.equal(revAdm1.reason, ADMISSION_REJECTION.UNSUPPORTED_REVERSED_DEPTH);

  // 4. camera.reversedDepthBuffer === true
  const revCam2 = createBasicCamera();
  revCam2.reversedDepthBuffer = true;
  const revAdm2 = canAdmitMesh(normalMesh, revCam2);
  assert.equal(revAdm2.admitted, false);
  assert.equal(revAdm2.reason, ADMISSION_REJECTION.UNSUPPORTED_REVERSED_DEPTH);

  // 5. Invalid depthFunc
  const badFuncMesh = createDepthTriangleMesh({ depthFunc: 999 });
  const badFuncAdm = canAdmitMesh(badFuncMesh, camera);
  assert.equal(badFuncAdm.admitted, false);
  assert.equal(badFuncAdm.reason, ADMISSION_REJECTION.INVALID_DEPTH_FUNC);
  assert.throws(
    () => extractMeshRenderData(badFuncMesh, camera, 64, 64),
    /Invalid or unsupported depthFunc/,
  );
});

test('Explicit Missing-Export Refusal: Mesh requiring depth strictly refuses Wasm lacking depth exports', () => {
  const mesh = createDepthTriangleMesh(); // depthTest=true, depthWrite=true
  const camera = createBasicCamera();

  // Mock Wasm that ONLY exposes legacy 8-arg exports
  let legacyCalled = false;
  const mockLegacyWasm = {
    f3d_build_mesh_packet: () => {
      legacyCalled = true;
      return new Uint8Array([1, 2, 3]);
    },
    f3d_build_canvas_mesh_packet: () => {
      legacyCalled = true;
      return new Uint8Array([4, 5, 6]);
    },
  };

  // 1. prepareMeshPacket must refuse when mesh requires depth
  assert.throws(
    () => prepareMeshPacket(mesh, camera, 64, 64, mockLegacyWasm),
    /Mesh requires depth \(depthTest or depthWrite enabled\), but wasmModule is missing f3d_build_mesh_depth_packet export/,
  );
  assert.equal(legacyCalled, false, 'Legacy export must NOT be silently called when depth is required');

  // 2. prepareCanvasMeshPacket must refuse when mesh requires depth
  assert.throws(
    () => prepareCanvasMeshPacket(mesh, camera, 64, 64, mockLegacyWasm),
    /Visible canvas mesh requires depth \(depthTest or depthWrite enabled\), but wasmModule is missing f3d_build_canvas_mesh_depth_packet export/,
  );
  assert.equal(legacyCalled, false, 'Canvas legacy export must NOT be silently called when depth is required');

  // 3. prepareMeshDepthPacket must refuse when depth export is missing
  assert.throws(
    () => prepareMeshDepthPacket(mesh, camera, 64, 64, mockLegacyWasm),
    /Mesh depth packet preparation failed: wasmModule is missing f3d_build_mesh_depth_packet/,
  );

  // 4. prepareCanvasMeshDepthPacket must refuse when canvas depth export is missing
  assert.throws(
    () => prepareCanvasMeshDepthPacket(mesh, camera, 64, 64, mockLegacyWasm),
    /Visible canvas mesh depth packet preparation failed: wasmModule is missing f3d_build_canvas_mesh_depth_packet/,
  );
});

test('Positive: prepareMeshDepthPacket and prepareCanvasMeshDepthPacket invoke Wasm with exact 11 typed arguments', () => {
  const mesh = createDepthTriangleMesh({ depthFunc: THREE.GreaterDepth, depthWrite: false });
  const camera = createBasicCamera();

  // 1. Offscreen depth packet with primary name
  let capturedOffscreen = null;
  const mockDepthWasm = {
    f3d_build_mesh_depth_packet: (pos, ind, mv, proj, col, w, h, webglDepth, depthTest, depthWrite, depthCompare) => {
      capturedOffscreen = { pos, ind, mv, proj, col, w, h, webglDepth, depthTest, depthWrite, depthCompare };
      return new Uint8Array([0x44, 0x45, 0x50, 0x54]); // 'DEPT'
    },
  };

  const offscreenResult = prepareMeshDepthPacket(mesh, camera, 320, 240, mockDepthWasm);
  assert.ok(capturedOffscreen, 'f3d_build_mesh_depth_packet must be invoked');
  assert.equal(capturedOffscreen.pos.length, 9);
  assert.equal(capturedOffscreen.ind.length, 0);
  assert.equal(capturedOffscreen.mv.length, 16);
  assert.equal(capturedOffscreen.proj.length, 16);
  assert.equal(capturedOffscreen.col.length, 4);
  assert.equal(capturedOffscreen.w, 320);
  assert.equal(capturedOffscreen.h, 240);
  assert.equal(capturedOffscreen.webglDepth, true);
  assert.equal(capturedOffscreen.depthTest, true);
  assert.equal(capturedOffscreen.depthWrite, false);
  assert.equal(capturedOffscreen.depthCompare, 5); // GreaterDepth -> wire 5
  assert.deepEqual(Array.from(offscreenResult.packetBytes), [0x44, 0x45, 0x50, 0x54]);

  // 2. Canvas depth packet with primary name
  let capturedCanvas = null;
  const mockCanvasDepthWasm = {
    f3d_build_canvas_mesh_depth_packet: (pos, ind, mv, proj, col, w, h, webglDepth, depthTest, depthWrite, depthCompare) => {
      capturedCanvas = { pos, ind, mv, proj, col, w, h, webglDepth, depthTest, depthWrite, depthCompare };
      return new Uint8Array([0x43, 0x44, 0x45, 0x50]); // 'CDEP'
    },
  };

  const canvasResult = prepareCanvasMeshDepthPacket(mesh, camera, 640, 480, mockCanvasDepthWasm);
  assert.ok(capturedCanvas, 'f3d_build_canvas_mesh_depth_packet must be invoked');
  assert.equal(capturedCanvas.w, 640);
  assert.equal(capturedCanvas.h, 480);
  assert.equal(capturedCanvas.depthTest, true);
  assert.equal(capturedCanvas.depthWrite, false);
  assert.equal(capturedCanvas.depthCompare, 5);
  assert.deepEqual(Array.from(canvasResult.packetBytes), [0x43, 0x44, 0x45, 0x50]);

  // 3. Fallback to canonical bridge aliases (gpu_bridge_build_mesh_depth_packet, gpu_bridge_build_canvas_mesh_depth_packet)
  let capturedBridgeAlias = null;
  const mockAliasWasm = {
    gpu_bridge_build_mesh_depth_packet: (...args) => {
      capturedBridgeAlias = args;
      return new Uint8Array([9, 9, 9]);
    },
  };
  const aliasResult = prepareMeshDepthPacket(mesh, camera, 100, 100, mockAliasWasm);
  assert.ok(capturedBridgeAlias);
  assert.equal(capturedBridgeAlias.length, 11);
  assert.deepEqual(Array.from(aliasResult.packetBytes), [9, 9, 9]);
});

test('Positive: Legacy no-depth mesh succeeds with 8-arg export when depth export absent, uses 11-arg when available', () => {
  const noDepthMesh = createBasicTriangleMesh({ depthTest: false, depthWrite: false });
  const camera = createBasicCamera();

  // 1. Only 8-arg export available -> succeeds via legacy 8-arg export
  let legacyCalled = false;
  let captured8Args = null;
  const mockLegacyWasm = {
    f3d_build_mesh_packet: (...args) => {
      legacyCalled = true;
      captured8Args = args;
      return new Uint8Array([8, 8, 8]);
    },
  };

  const resLegacy = prepareMeshPacket(noDepthMesh, camera, 64, 64, mockLegacyWasm);
  assert.equal(legacyCalled, true);
  assert.equal(captured8Args.length, 8);
  assert.deepEqual(Array.from(resLegacy.packetBytes), [8, 8, 8]);

  // 2. 11-arg export available -> uses 11-arg export with depthTest=false, depthWrite=false, depthCompare=ALWAYS (8)
  let depthCalled = false;
  let captured11Args = null;
  const mockDepthWasm = {
    f3d_build_mesh_depth_packet: (...args) => {
      depthCalled = true;
      captured11Args = args;
      return new Uint8Array([11, 11, 11]);
    },
    f3d_build_mesh_packet: () => {
      throw new Error('Should prefer depth export when available');
    },
  };

  const resDepth = prepareMeshPacket(noDepthMesh, camera, 64, 64, mockDepthWasm);
  assert.equal(depthCalled, true);
  assert.equal(captured11Args.length, 11);
  assert.equal(captured11Args[8], false); // depthTest
  assert.equal(captured11Args[9], false); // depthWrite
  assert.equal(captured11Args[10], 8);    // depthCompare ALWAYS
  assert.deepEqual(Array.from(resDepth.packetBytes), [11, 11, 11]);
});

test('Positive: renderMesh routes depth-enabled mesh through depth packet builder to WebGpuBridgeHost', async () => {
  const mesh = createDepthTriangleMesh({ depthFunc: THREE.LessDepth });
  const camera = createBasicCamera();

  let executedPacket = null;
  let executedContext = null;
  const mockBridgeHost = {
    executePacket: async (packet, ctx) => {
      executedPacket = packet;
      executedContext = ctx;
      return { readbackBufferId: 42 };
    },
  };

  const mockWasm = {
    f3d_build_mesh_depth_packet: (pos, ind, mv, proj, col, w, h, webglDepth, dt, dw, dc) => {
      assert.equal(dt, true);
      assert.equal(dw, true);
      assert.equal(dc, 2); // LessDepth -> 2
      return new Uint8Array([0x52, 0x45, 0x4e, 0x44]); // 'REND'
    },
    f3d_build_canvas_mesh_depth_packet: (pos, ind, mv, proj, col, w, h, webglDepth, dt, dw, dc) => {
      assert.equal(dt, true);
      assert.equal(dw, true);
      assert.equal(dc, 2);
      return new Uint8Array([0x43, 0x52, 0x45, 0x4e]); // 'CREN'
    },
  };

  // 1. Offscreen renderMesh with depth
  const offscreenRes = await renderMesh(mockBridgeHost, mesh, camera, null, mockWasm, { width: 64, height: 64 });
  assert.equal(offscreenRes.target, 'offscreen');
  assert.equal(executedContext, null);
  assert.deepEqual(Array.from(executedPacket), [0x52, 0x45, 0x4e, 0x44]);
  assert.equal(offscreenRes.snapshot.depthCompare, 2);

  // 2. Canvas renderMesh with depth
  const mockCanvasContext = { canvas: { width: 128, height: 128 } };
  const canvasRes = await renderMesh(mockBridgeHost, mesh, camera, mockCanvasContext, mockWasm);
  assert.equal(canvasRes.target, 'canvas');
  assert.equal(executedContext, mockCanvasContext);
  assert.deepEqual(Array.from(executedPacket), [0x43, 0x52, 0x45, 0x4e]);
  assert.equal(canvasRes.snapshot.depthCompare, 2);
});

test('Negative: prepareMeshBatchPacket rejects empty or non-array mesh inputs with EMPTY_MESH_BATCH', () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array(0) };

  assert.throws(
    () => prepareMeshBatchPacket([], camera, 64, 64, mockWasm),
    new RegExp(ADMISSION_REJECTION.EMPTY_MESH_BATCH)
  );

  assert.throws(
    () => prepareMeshBatchPacket(null, camera, 64, 64, mockWasm),
    new RegExp(ADMISSION_REJECTION.EMPTY_MESH_BATCH)
  );

  assert.throws(
    () => prepareMeshBatchPacket('not-an-array', camera, 64, 64, mockWasm),
    new RegExp(ADMISSION_REJECTION.EMPTY_MESH_BATCH)
  );
});

test('Negative: prepareMeshBatchPacket rejects wasmModule missing f3d_build_mesh_batch_packet export', () => {
  const mesh = createBasicTriangleMesh();
  const camera = createBasicCamera();

  assert.throws(
    () => prepareMeshBatchPacket([mesh], camera, 64, 64, {}),
    /missing f3d_build_mesh_batch_packet/
  );

  assert.throws(
    () => prepareMeshBatchPacket([mesh], camera, 64, 64, null),
    /missing f3d_build_mesh_batch_packet/
  );
});

test('Negative: prepareMeshBatchPacket refuses incompatible shared depth settings across meshes', () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array(0) };

  // 1. Incompatible depthTest
  const meshA = createDepthTriangleMesh({ depthTest: true, depthWrite: true });
  const meshB = createDepthTriangleMesh({ depthTest: false, depthWrite: false });
  assert.throws(
    () => prepareMeshBatchPacket([meshA, meshB], camera, 64, 64, mockWasm),
    (err) => err.message.includes(ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH) && err.message.includes('depthTest')
  );

  // 2. Incompatible depthWrite
  const meshC = createDepthTriangleMesh({ depthTest: true, depthWrite: true });
  const meshD = createDepthTriangleMesh({ depthTest: true, depthWrite: false });
  assert.throws(
    () => prepareMeshBatchPacket([meshC, meshD], camera, 64, 64, mockWasm),
    (err) => err.message.includes(ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH) && err.message.includes('depthWrite')
  );

  // 3. Incompatible depthCompare (depthFunc)
  const meshE = createDepthTriangleMesh({ depthTest: true, depthWrite: true, depthFunc: THREE.LessDepth });
  const meshF = createDepthTriangleMesh({ depthTest: true, depthWrite: true, depthFunc: THREE.GreaterDepth });
  assert.throws(
    () => prepareMeshBatchPacket([meshE, meshF], camera, 64, 64, mockWasm),
    (err) => err.message.includes(ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH) && err.message.includes('depthCompare')
  );
});

test('Negative: prepareMeshBatchPacket rejects batch containing an inadmissible mesh with exact index', () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array(0) };

  const validMesh = createBasicTriangleMesh();
  const invisibleMesh = createBasicTriangleMesh();
  invisibleMesh.visible = false;

  assert.throws(
    () => prepareMeshBatchPacket([validMesh, invisibleMesh], camera, 64, 64, mockWasm),
    (err) => err.message.includes('Mesh batch admission rejected at index 1') && err.message.includes(ADMISSION_REJECTION.NOT_VISIBLE)
  );

  const transparentMesh = createBasicTriangleMesh({ transparent: true });
  assert.throws(
    () => prepareMeshBatchPacket([transparentMesh, validMesh], camera, 64, 64, mockWasm),
    (err) => err.message.includes('Mesh batch admission rejected at index 0') && err.message.includes(ADMISSION_REJECTION.UNSUPPORTED_MATERIAL)
  );
});

test('Positive: prepareMeshBatchPacket flattens expanded positions and packs per-draw uniforms for variable-length mesh batch', () => {
  const camera = createBasicCamera();

  // Mesh 1: Unindexed Triangle (3 vertices)
  const mesh1 = createBasicTriangleMesh({ color: 0xff0000 });
  mesh1.position.set(1, 0, 0);
  mesh1.updateMatrixWorld();

  // Mesh 2: Indexed Quad (4 vertices, 6 indices -> 6 expanded vertices)
  const quadGeom = new THREE.BufferGeometry();
  quadGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]), 3));
  quadGeom.setIndex([0, 1, 2, 0, 2, 3]);
  const quadMat = new THREE.MeshBasicMaterial({
    color: 0x00ff00,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh2 = new THREE.Mesh(quadGeom, quadMat);
  mesh2.position.set(5, 0, 0);
  mesh2.updateMatrixWorld();

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x42, 0x41, 0x54, 0x43, 0x48]); // 'BATCH'
    },
  };

  // 1. Offscreen batch
  const batchRes = prepareMeshBatchPacket([mesh1, mesh2], camera, 64, 64, mockWasm);
  assert.equal(batchRes.meshCount, 2);
  assert.equal(batchRes.totalVertices, 3 + 6);
  assert.equal(batchRes.target, 'offscreen');
  assert.deepEqual(Array.from(batchRes.packetBytes), [0x42, 0x41, 0x54, 0x43, 0x48]);

  // Inspect captured arguments
  assert.ok(capturedArgs.flatPos instanceof Float32Array);
  assert.equal(capturedArgs.flatPos.length, (3 + 6) * 3); // 27 floats

  assert.ok(capturedArgs.vCounts instanceof Uint32Array);
  assert.equal(capturedArgs.vCounts.length, 2);
  assert.equal(capturedArgs.vCounts[0], 3);
  assert.equal(capturedArgs.vCounts[1], 6);

  assert.ok(capturedArgs.mvs instanceof Float64Array);
  assert.equal(capturedArgs.mvs.length, 2 * 16); // 32 floats
  // Verify distinct model-view matrices (mesh1 x=1 vs mesh2 x=5)
  assert.notEqual(capturedArgs.mvs[12], capturedArgs.mvs[16 + 12]);

  assert.ok(capturedArgs.cols instanceof Float32Array);
  assert.equal(capturedArgs.cols.length, 2 * 4); // 8 floats
  // Mesh 1: Red (1, 0, 0, 1)
  assert.equal(capturedArgs.cols[0], 1.0);
  assert.equal(capturedArgs.cols[1], 0.0);
  assert.equal(capturedArgs.cols[2], 0.0);
  assert.equal(capturedArgs.cols[3], 1.0);
  // Mesh 2: Green (0, 1, 0, 1)
  assert.equal(capturedArgs.cols[4], 0.0);
  assert.equal(capturedArgs.cols[5], 1.0);
  assert.equal(capturedArgs.cols[6], 0.0);
  assert.equal(capturedArgs.cols[7], 1.0);

  assert.ok(capturedArgs.proj instanceof Float64Array);
  assert.equal(capturedArgs.proj.length, 16);

  assert.equal(capturedArgs.w, 64);
  assert.equal(capturedArgs.h, 64);
  assert.equal(capturedArgs.webglDepth, true);
  assert.equal(capturedArgs.dt, false);
  assert.equal(capturedArgs.dw, false);
  assert.equal(capturedArgs.dc, 8); // AlwaysDepth wire code for depthTest=false
  assert.equal(capturedArgs.canvas, false);

  // 2. Canvas batch
  const canvasBatchRes = prepareMeshBatchPacket([mesh1, mesh2], camera, 128, 128, mockWasm, { target: 'canvas' });
  assert.equal(canvasBatchRes.target, 'canvas');
  assert.equal(capturedArgs.canvas, true);
  assert.equal(capturedArgs.w, 128);
  assert.equal(capturedArgs.h, 128);
});

test('Positive: renderMeshBatch routes variable-length batch through WebGpuBridgeHost for canvas and offscreen', async () => {
  const camera = createBasicCamera();
  const mesh1 = createBasicTriangleMesh({ color: 0xff0000 });
  const mesh2 = createBasicTriangleMesh({ color: 0x0000ff });

  let executedPacket = null;
  let executedContext = null;
  const mockBridgeHost = {
    executePacket: async (packet, ctx) => {
      executedPacket = packet;
      executedContext = ctx;
      return { status: 'BATCH_OK' };
    },
  };

  const mockWasm = {
    f3d_build_mesh_batch_packet: () => new Uint8Array([0x99, 0x88, 0x77]),
  };

  // 1. Offscreen renderMeshBatch
  const offscreenRes = await renderMeshBatch(mockBridgeHost, [mesh1, mesh2], camera, null, mockWasm, { width: 64, height: 64 });
  assert.equal(offscreenRes.target, 'offscreen');
  assert.equal(offscreenRes.meshCount, 2);
  assert.equal(executedContext, null);
  assert.deepEqual(Array.from(executedPacket), [0x99, 0x88, 0x77]);

  // 2. Canvas renderMeshBatch
  const mockCanvasContext = { canvas: { width: 256, height: 256 } };
  const canvasRes = await renderMeshBatch(mockBridgeHost, [mesh1, mesh2], camera, mockCanvasContext, mockWasm);
  assert.equal(canvasRes.target, 'canvas');
  assert.equal(canvasRes.meshCount, 2);
  assert.equal(executedContext, mockCanvasContext);
  assert.deepEqual(Array.from(executedPacket), [0x99, 0x88, 0x77]);
});

test('Regression: prepareMeshBatchPacket and renderMeshBatch honor explicit target and do not let options.canvas leak into offscreen', async () => {
  const camera = createBasicCamera();
  const mesh1 = createBasicTriangleMesh({ color: 0xff0000 });
  const mesh2 = createBasicTriangleMesh({ color: 0x00ff00 });

  let capturedCanvasFlag = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedCanvasFlag = canvas;
      return new Uint8Array([0x11, 0x22]);
    },
  };

  // 1. Calling prepareMeshBatchPacket with options = { canvas: true } without target: 'canvas' must emit canvas = false
  const res1 = prepareMeshBatchPacket([mesh1, mesh2], camera, 64, 64, mockWasm, { canvas: true });
  assert.equal(res1.target, 'offscreen');
  assert.equal(capturedCanvasFlag, false, 'options.canvas must NOT override target; target must be explicit');

  // 2. Calling renderMeshBatch with canvasContext = null and options = { canvas: true } must strictly remain offscreen
  let executedContext = undefined;
  const mockBridgeHost = {
    executePacket: async (packet, ctx) => {
      executedContext = ctx;
      return { ok: true };
    },
  };
  const res2 = await renderMeshBatch(mockBridgeHost, [mesh1, mesh2], camera, null, mockWasm, { canvas: true });
  assert.equal(res2.target, 'offscreen');
  assert.equal(capturedCanvasFlag, false);
  assert.equal(executedContext, null);
});

test('Regression: renderMesh and renderMeshBatch propagate width: 0 and height: 0 to validation via ?? rather than defaulting to 64', async () => {
  const camera = createBasicCamera();
  const mesh = createBasicTriangleMesh();
  const mockWasm = {
    f3d_build_mesh_packet: () => new Uint8Array(0),
    f3d_build_mesh_batch_packet: () => new Uint8Array(0),
  };
  const mockBridgeHost = { executePacket: async () => ({}) };

  // 1. renderMesh with width: 0 must reject with INVALID_DIMENSIONS, not silently become 64
  await assert.rejects(
    () => renderMesh(mockBridgeHost, mesh, camera, null, mockWasm, { width: 0, height: 64 }),
    new RegExp(ADMISSION_REJECTION.INVALID_DIMENSIONS)
  );
  await assert.rejects(
    () => renderMesh(mockBridgeHost, mesh, camera, null, mockWasm, { width: 64, height: 0 }),
    new RegExp(ADMISSION_REJECTION.INVALID_DIMENSIONS)
  );

  // 2. renderMeshBatch with width: 0 must reject with INVALID_DIMENSIONS
  await assert.rejects(
    () => renderMeshBatch(mockBridgeHost, [mesh], camera, null, mockWasm, { width: 0, height: 64 }),
    new RegExp(ADMISSION_REJECTION.INVALID_DIMENSIONS)
  );
  await assert.rejects(
    () => renderMeshBatch(mockBridgeHost, [mesh], camera, null, mockWasm, { width: 64, height: 0 }),
    new RegExp(ADMISSION_REJECTION.INVALID_DIMENSIONS)
  );
});

test('Positive: prepareMeshBatchPacket preserves zero-drawrange mesh without losing or miscounting batch draw slots (empty prefix and middle)', async () => {
  const camera = createBasicCamera();

  // Mesh 0: Empty unindexed prefix mesh (drawRange count 0)
  const mesh0 = createBasicTriangleMesh({ color: 0xff0000 });
  mesh0.geometry.setDrawRange(0, 0);
  mesh0.position.set(-5, 0, 0);
  mesh0.updateMatrixWorld();

  // Mesh 1: Valid unindexed triangle (3 vertices)
  const mesh1 = createBasicTriangleMesh({ color: 0x00ff00 });
  mesh1.position.set(0, 0, 0);
  mesh1.updateMatrixWorld();

  // Mesh 2: Empty indexed middle mesh (drawRange count 0)
  const quadGeom = new THREE.BufferGeometry();
  quadGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]), 3));
  quadGeom.setIndex([0, 1, 2, 0, 2, 3]);
  quadGeom.setDrawRange(2, 0); // Empty range
  const quadMat = new THREE.MeshBasicMaterial({
    color: 0x0000ff,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh2 = new THREE.Mesh(quadGeom, quadMat);
  mesh2.position.set(5, 0, 0);
  mesh2.updateMatrixWorld();

  // Mesh 3: Valid unindexed triangle (3 vertices)
  const mesh3 = createBasicTriangleMesh({ color: 0xffff00 });
  mesh3.position.set(10, 0, 0);
  mesh3.updateMatrixWorld();

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x5a, 0x45, 0x52, 0x4f]); // 'ZERO'
    },
  };

  const meshes = [mesh0, mesh1, mesh2, mesh3];
  const batchRes = prepareMeshBatchPacket(meshes, camera, 64, 64, mockWasm);

  // 1. Overall batch counts: exactly 4 meshes, total 6 vertices (0 + 3 + 0 + 3)
  assert.equal(batchRes.meshCount, 4);
  assert.equal(batchRes.totalVertices, 6);
  assert.equal(batchRes.snapshots.length, 4);

  // 2. Snapshot vertex counts
  assert.equal(batchRes.snapshots[0].vertexCount, 0);
  assert.equal(batchRes.snapshots[0].expandedPositions.length, 0);
  assert.equal(batchRes.snapshots[1].vertexCount, 3);
  assert.equal(batchRes.snapshots[1].expandedPositions.length, 9);
  assert.equal(batchRes.snapshots[2].vertexCount, 0);
  assert.equal(batchRes.snapshots[2].expandedPositions.length, 0);
  assert.equal(batchRes.snapshots[3].vertexCount, 3);
  assert.equal(batchRes.snapshots[3].expandedPositions.length, 9);

  // 3. Captured vertex_counts: exactly [0, 3, 0, 3] without dropping empty slots
  assert.ok(capturedArgs.vCounts instanceof Uint32Array);
  assert.equal(capturedArgs.vCounts.length, 4);
  assert.equal(capturedArgs.vCounts[0], 0);
  assert.equal(capturedArgs.vCounts[1], 3);
  assert.equal(capturedArgs.vCounts[2], 0);
  assert.equal(capturedArgs.vCounts[3], 3);

  // 4. Flat positions: exactly 18 floats (6 vertices * 3) without placeholder shift
  assert.ok(capturedArgs.flatPos instanceof Float32Array);
  assert.equal(capturedArgs.flatPos.length, 18);
  // Mesh 1 positions start at index 0
  assert.deepEqual(capturedArgs.flatPos.slice(0, 9), batchRes.snapshots[1].expandedPositions);
  // Mesh 3 positions start at index 9 (immediately following mesh 1, mesh 2 has 0 vertices)
  assert.deepEqual(capturedArgs.flatPos.slice(9, 18), batchRes.snapshots[3].expandedPositions);

  // 5. Model-views: exactly 4 * 16 = 64 floats, preserving all 4 draw transforms in order
  assert.ok(capturedArgs.mvs instanceof Float64Array);
  assert.equal(capturedArgs.mvs.length, 4 * 16);
  assert.deepEqual(capturedArgs.mvs.slice(0, 16), batchRes.snapshots[0].modelView);
  assert.deepEqual(capturedArgs.mvs.slice(16, 32), batchRes.snapshots[1].modelView);
  assert.deepEqual(capturedArgs.mvs.slice(32, 48), batchRes.snapshots[2].modelView);
  assert.deepEqual(capturedArgs.mvs.slice(48, 64), batchRes.snapshots[3].modelView);

  // 6. Colors: exactly 4 * 4 = 16 floats, preserving all 4 draw colors in order
  assert.ok(capturedArgs.cols instanceof Float32Array);
  assert.equal(capturedArgs.cols.length, 4 * 4);
  // Mesh 0: Red
  assert.equal(capturedArgs.cols[0], 1.0);
  assert.equal(capturedArgs.cols[1], 0.0);
  // Mesh 1: Green
  assert.equal(capturedArgs.cols[4], 0.0);
  assert.equal(capturedArgs.cols[5], 1.0);
  // Mesh 2: Blue
  assert.equal(capturedArgs.cols[8], 0.0);
  assert.equal(capturedArgs.cols[10], 1.0);
  // Mesh 3: Yellow
  assert.equal(capturedArgs.cols[12], 1.0);
  assert.equal(capturedArgs.cols[13], 1.0);

  // 7. renderMeshBatch executes successfully with empty drawRange meshes
  let executedPacket = null;
  const mockBridgeHost = {
    executePacket: async (packet) => {
      executedPacket = packet;
      return { status: 'BATCH_ZERO_OK' };
    },
  };
  const execRes = await renderMeshBatch(mockBridgeHost, meshes, camera, null, mockWasm, { width: 64, height: 64 });
  assert.equal(execRes.meshCount, 4);
  assert.equal(execRes.totalVertices, 6);
  assert.deepEqual(Array.from(executedPacket), [0x5a, 0x45, 0x52, 0x4f]);
});

test('Positive: renderScene collects 3 meshes with distinct colors/transforms into one batch with 3 draw ranges and correct MV per mesh', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const mesh1 = createBasicTriangleMesh({ color: 0xff0000 });
  mesh1.position.set(1, 2, 3);
  mesh1.rotation.set(0, Math.PI / 4, 0);

  const mesh2 = createBasicTriangleMesh({ color: 0x00ff00 });
  mesh2.position.set(-2, 0, 1);
  mesh2.scale.set(2, 2, 2);

  const mesh3 = createBasicTriangleMesh({ color: 0x0000ff });
  mesh3.position.set(0, -3, -5);

  scene.add(mesh1);
  scene.add(mesh2);
  scene.add(mesh3);

  let capturedBatchArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedBatchArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x53, 0x43, 0x45, 0x4e, 0x45]); // 'SCENE'
    },
  };

  let executedPacket = null;
  let executedTarget = null;
  const mockBridgeHost = {
    executePacket: async (packet, target) => {
      executedPacket = packet;
      executedTarget = target;
      return { status: 'SCENE_RENDERED' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // 1. Returned shape has admitted and refused
  assert.deepEqual(res.admitted, [mesh1.uuid, mesh2.uuid, mesh3.uuid]);
  assert.deepEqual(res.refused, []);
  assert.equal(res.result?.status, 'SCENE_RENDERED');

  // 2. Exactly one batch packet executed
  assert.deepEqual(Array.from(executedPacket), [0x53, 0x43, 0x45, 0x4e, 0x45]);
  assert.equal(executedTarget, null);

  // 3. Exactly 3 draw ranges
  assert.equal(capturedBatchArgs.vCounts.length, 3);
  assert.equal(capturedBatchArgs.vCounts[0], 3);
  assert.equal(capturedBatchArgs.vCounts[1], 3);
  assert.equal(capturedBatchArgs.vCounts[2], 3);

  // 4. Distinct colors per draw (mesh 1: red, mesh 2: green, mesh 3: blue)
  assert.equal(capturedBatchArgs.cols.length, 12);
  // Mesh 1: Red [1, 0, 0, 1]
  assert.equal(capturedBatchArgs.cols[0], 1);
  assert.equal(capturedBatchArgs.cols[1], 0);
  assert.equal(capturedBatchArgs.cols[2], 0);
  // Mesh 2: Green [0, 1, 0, 1]
  assert.equal(capturedBatchArgs.cols[4], 0);
  assert.equal(capturedBatchArgs.cols[5], 1);
  assert.equal(capturedBatchArgs.cols[6], 0);
  // Mesh 3: Blue [0, 0, 1, 1]
  assert.equal(capturedBatchArgs.cols[8], 0);
  assert.equal(capturedBatchArgs.cols[9], 0);
  assert.equal(capturedBatchArgs.cols[10], 1);

  // 5. Correct MV per mesh computed via camera.matrixWorldInverse * mesh.matrixWorld
  assert.equal(capturedBatchArgs.mvs.length, 48);
  const expectedMv1 = multiplyMatrices4x4(camera.matrixWorldInverse, mesh1.matrixWorld);
  const expectedMv2 = multiplyMatrices4x4(camera.matrixWorldInverse, mesh2.matrixWorld);
  const expectedMv3 = multiplyMatrices4x4(camera.matrixWorldInverse, mesh3.matrixWorld);
  assert.deepEqual(capturedBatchArgs.mvs.slice(0, 16), expectedMv1);
  assert.deepEqual(capturedBatchArgs.mvs.slice(16, 32), expectedMv2);
  assert.deepEqual(capturedBatchArgs.mvs.slice(32, 48), expectedMv3);
});

test('Refusal: renderScene refuses WHOLE submission when visible InstancedMesh or unsupported-side mesh exists (no partial render)', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const validMesh1 = createBasicTriangleMesh({ color: 0xff0000 });

  // InstancedMesh: unsupported subclass
  const instGeom = new THREE.BufferGeometry();
  instGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const instMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, depthTest: false, depthWrite: false });
  const instancedMesh = new THREE.InstancedMesh(instGeom, instMat, 2);

  // Unsupported side: side 99
  const invalidSideMesh = createBasicTriangleMesh({ side: 99 });

  const validMesh2 = createBasicTriangleMesh({ color: 0x00ff00 });

  scene.add(validMesh1);
  scene.add(instancedMesh);
  scene.add(invalidSideMesh);
  scene.add(validMesh2);

  let capturedBatchArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedBatchArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x54, 0x57, 0x4f]); // 'TWO'
    },
  };

  let executionCount = 0;
  const mockBridgeHost = {
    executePacket: async () => {
      executionCount++;
      return { status: 'PARTIAL_OK' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // Whole submission refused: admitted is empty (Root Mail 14355: no partial scene rendering)
  assert.deepEqual(res.admitted, []);

  // Refused contains exactly the 2 invalid meshes with reasons
  assert.equal(res.refused.length, 2);
  assert.deepEqual(res.refused[0], {
    uuid: instancedMesh.uuid,
    reason: ADMISSION_REJECTION.UNSUPPORTED_MESH_SUBCLASS,
  });
  assert.deepEqual(res.refused[1], {
    uuid: invalidSideMesh.uuid,
    reason: ADMISSION_REJECTION.UNSUPPORTED_SIDE,
  });

  // Strict refusal contract: NO batch packet executed, NO batch packet prepared
  assert.equal(executionCount, 0, 'No packet may be submitted when scene contains visible unsupported items');
  assert.equal(capturedBatchArgs, null, 'No batch packet builder may be invoked when scene contains refusals');
  assert.ok(res.reason.includes('UNSUPPORTED_MESH_SUBCLASS') || res.refusalReason.includes('UNSUPPORTED_MESH_SUBCLASS'));

  // Retain renderMeshBatch as explicit API for caller-selected subsets
  const batchRes = await renderMeshBatch(mockBridgeHost, [validMesh1, validMesh2], camera, null, mockWasm);
  assert.equal(executionCount, 1, 'renderMeshBatch permits caller-selected subset execution');
  assert.equal(batchRes.meshCount, 2);
  assert.equal(capturedBatchArgs.vCounts.length, 2);
});

test('Positive: renderScene propagates nested Group transforms into emitted per-mesh model-view matrix via single updateMatrixWorld', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const rootGroup = new THREE.Group();
  rootGroup.position.set(10, 0, 0);

  const childGroup = new THREE.Group();
  childGroup.position.set(0, 5, 0);
  childGroup.rotation.set(0, 0, Math.PI / 2);

  const nestedMesh = createBasicTriangleMesh({ color: 0x123456 });
  nestedMesh.position.set(1, 2, 3);

  childGroup.add(nestedMesh);
  rootGroup.add(childGroup);
  scene.add(rootGroup);

  let capturedBatchArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedBatchArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x47, 0x52, 0x50]); // 'GRP'
    },
  };

  const mockBridgeHost = {
    executePacket: async () => ({ status: 'NESTED_OK' }),
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  assert.deepEqual(res.admitted, [nestedMesh.uuid]);
  assert.deepEqual(res.refused, []);

  // Verify nested transform in world space:
  // Position (1, 2, 3) rotated by 90 deg around Z -> (-2, 1, 3)
  // + childGroup (0, 5, 0) -> (-2, 6, 3)
  // + rootGroup (10, 0, 0) -> (8, 6, 3)
  assert.ok(Math.abs(nestedMesh.matrixWorld.elements[12] - 8) < 1e-5);
  assert.ok(Math.abs(nestedMesh.matrixWorld.elements[13] - 6) < 1e-5);
  assert.ok(Math.abs(nestedMesh.matrixWorld.elements[14] - 3) < 1e-5);

  const expectedMv = multiplyMatrices4x4(camera.matrixWorldInverse, nestedMesh.matrixWorld);
  assert.deepEqual(capturedBatchArgs.mvs, expectedMv);
});

test('Negative: renderScene with empty admitted set explicitly refuses without submitting', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  // Add only inadmissible meshes
  const instGeom = new THREE.BufferGeometry();
  instGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const instMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  const instMesh = new THREE.InstancedMesh(instGeom, instMat, 1);

  const invalidMesh = createBasicTriangleMesh({ side: 99 });

  scene.add(instMesh);
  scene.add(invalidMesh);

  let executeCalled = false;
  const mockBridgeHost = {
    executePacket: async () => {
      executeCalled = true;
      return { status: 'SHOULD_NOT_EXECUTE' };
    },
  };

  const mockWasm = {
    f3d_build_mesh_batch_packet: () => {
      throw new Error('Should not build batch for empty admitted set');
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  assert.deepEqual(res.admitted, []);
  assert.equal(res.refused.length, 2);
  assert.equal(res.refused[0].uuid, instMesh.uuid);
  assert.equal(res.refused[1].uuid, invalidMesh.uuid);
  assert.equal(executeCalled, false);
});

test('Positive: renderScene routes to canvas swapchain when canvasContext is provided', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();
  const mesh = createBasicTriangleMesh();
  scene.add(mesh);

  let capturedCanvasArg = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedCanvasArg = canvas;
      return new Uint8Array([0x43, 0x41, 0x4e]); // 'CAN'
    },
  };

  const fakeCanvasContext = {
    canvas: { width: 320, height: 240 },
  };

  let targetPassed = null;
  const mockBridgeHost = {
    executePacket: async (packet, target) => {
      targetPassed = target;
      return { status: 'CANVAS_OK' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, fakeCanvasContext, mockWasm);

  assert.deepEqual(res.admitted, [mesh.uuid]);
  assert.equal(capturedCanvasArg, true);
  assert.equal(targetPassed, fakeCanvasContext);
});

test('Negative: renderScene validates required arguments (bridgeHost, scene, camera)', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();
  const mockBridgeHost = { executePacket: async () => {} };

  // Missing bridgeHost
  await assert.rejects(
    () => renderScene(null, scene, camera, null, {}),
    /Invalid bridgeHost/
  );

  // Missing scene
  await assert.rejects(
    () => renderScene(mockBridgeHost, null, camera, null, {}),
    /Invalid scene/
  );

  // Missing camera
  await assert.rejects(
    () => renderScene(mockBridgeHost, scene, null, null, {}),
    new RegExp(ADMISSION_REJECTION.INVALID_CAMERA)
  );
});

test('Regression: prepareMeshBatchPacket, renderMeshBatch, and renderScene map empty batch refusals to EMPTY_MESH_BATCH code', async () => {
  const camera = createBasicCamera();
  const mockBridgeHost = { executePacket: async () => ({ status: 'OK' }) };
  let wasmCalled = false;
  const mockWasm = {
    f3d_build_mesh_batch_packet: () => {
      wasmCalled = true;
      return new Uint8Array(0);
    },
  };

  // 1. prepareMeshBatchPacket refuses before calling Wasm and contains EMPTY_MESH_BATCH code + err.reason
  let err1 = null;
  try {
    prepareMeshBatchPacket([], camera, 64, 64, mockWasm);
  } catch (err) {
    err1 = err;
  }
  assert.ok(err1, 'prepareMeshBatchPacket must throw on empty batch');
  assert.ok(
    err1.message.startsWith('EMPTY_MESH_BATCH:'),
    `Error message "${err1.message}" must be prefixed with EMPTY_MESH_BATCH:`
  );
  assert.equal(err1.reason, 'EMPTY_MESH_BATCH', 'err.reason must be set to EMPTY_MESH_BATCH');
  assert.equal(wasmCalled, false, 'Wasm must not be called on empty batch');

  // 2. renderMeshBatch refuses before calling Wasm or bridgeHost and contains EMPTY_MESH_BATCH code + err.reason
  let err2 = null;
  try {
    await renderMeshBatch(mockBridgeHost, [], camera, null, mockWasm);
  } catch (err) {
    err2 = err;
  }
  assert.ok(err2, 'renderMeshBatch must throw on empty batch');
  assert.ok(
    err2.message.startsWith('EMPTY_MESH_BATCH:'),
    `renderMeshBatch error message "${err2.message}" must be prefixed with EMPTY_MESH_BATCH:`
  );
  assert.equal(err2.reason, 'EMPTY_MESH_BATCH', 'renderMeshBatch err.reason must be EMPTY_MESH_BATCH');

  // 3. If Wasm itself throws Rust MeshPacketError::EmptyMeshList, prepareMeshBatchPacket maps it to EMPTY_MESH_BATCH
  const mesh = createBasicTriangleMesh();
  const mockWasmEmptyError = {
    f3d_build_mesh_batch_packet: () => {
      throw new Error('mesh inputs list must contain at least one mesh');
    },
  };
  let err3 = null;
  try {
    prepareMeshBatchPacket([mesh], camera, 64, 64, mockWasmEmptyError);
  } catch (err) {
    err3 = err;
  }
  assert.ok(err3, 'prepareMeshBatchPacket must map Wasm empty error');
  assert.ok(
    err3.message.startsWith('EMPTY_MESH_BATCH:'),
    `Wasm EmptyMeshList must be mapped to EMPTY_MESH_BATCH prefix (got: "${err3.message}")`
  );
  assert.equal(err3.reason, 'EMPTY_MESH_BATCH', 'Wasm EmptyMeshList mapped error must have err.reason');

  // 4. renderScene returns refusal reason containing EMPTY_MESH_BATCH for empty scene
  const emptyScene = new THREE.Scene();
  const res = await renderScene(mockBridgeHost, emptyScene, camera, null, mockWasm);
  assert.deepEqual(res.admitted, []);
  assert.equal(res.reason, 'EMPTY_MESH_BATCH');
  assert.equal(res.refusalReason, 'EMPTY_MESH_BATCH');
});

test('Contract: All ADMISSION_REJECTION thrown errors carry prefix with key code and err.reason = key', () => {
  const camera = createBasicCamera();
  const mesh = createBasicTriangleMesh();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array(0) };

  // 1. Every entry in ADMISSION_REJECTION must be prefixed with its own key name + ': '
  for (const [key, value] of Object.entries(ADMISSION_REJECTION)) {
    assert(
      value.startsWith(`${key}: `),
      `ADMISSION_REJECTION.${key} must start with "${key}: ", got: "${value}"`
    );
    const err = createAdmissionError(key);
    assert.equal(err.reason, key);
    assert(
      err.message.startsWith(`${key}: `),
      `createAdmissionError(${key}).message must start with "${key}: ", got: "${err.message}"`
    );
  }

  // 2. EMPTY_MESH_BATCH thrown from prepareMeshBatchPacket
  assert.throws(
    () => prepareMeshBatchPacket([], camera, 64, 64, mockWasm),
    (err) => err.message.startsWith('EMPTY_MESH_BATCH:') && err.reason === 'EMPTY_MESH_BATCH'
  );

  // 3. INVALID_DIMENSIONS thrown from extractMeshRenderData
  assert.throws(
    () => extractMeshRenderData(mesh, camera, 0, 64),
    (err) => err.message.startsWith('INVALID_DIMENSIONS:') && err.reason === 'INVALID_DIMENSIONS'
  );

  // 4. INVALID_DRAWRANGE thrown from extractMeshRenderData
  mesh.geometry.setDrawRange(-1, 3);
  assert.throws(
    () => extractMeshRenderData(mesh, camera, 64, 64),
    (err) => err.message.startsWith('INVALID_DRAWRANGE:') && err.reason === 'INVALID_DRAWRANGE'
  );
  mesh.geometry.setDrawRange(0, Infinity); // restore

  // 5. Single-mesh admission refusal thrown from extractMeshRenderData
  const invisibleSingle = createBasicTriangleMesh();
  invisibleSingle.visible = false;
  assert.throws(
    () => extractMeshRenderData(invisibleSingle, camera, 64, 64),
    (err) => err.message.startsWith('NOT_VISIBLE:') && err.reason === 'NOT_VISIBLE'
  );

  const notAMesh = { isMesh: false, geometry: {}, material: {} };
  assert.throws(
    () => extractMeshRenderData(notAMesh, camera, 64, 64),
    (err) => err.message.startsWith('NOT_A_MESH:') && err.reason === 'NOT_A_MESH'
  );

  // 6. AMBIGUOUS_DEPTH_PAIR thrown from extractMeshRenderData
  const ambigMesh = createBasicTriangleMesh({ depthTest: false, depthWrite: true });
  assert.throws(
    () => extractMeshRenderData(ambigMesh, camera, 64, 64),
    (err) => err.message.startsWith('AMBIGUOUS_DEPTH_PAIR:') && err.reason === 'AMBIGUOUS_DEPTH_PAIR'
  );

  // 7. INCOMPATIBLE_BATCH_DEPTH thrown from prepareMeshBatchPacket
  const meshA = createBasicTriangleMesh({ depthTest: true });
  const meshB = createBasicTriangleMesh({ depthTest: false });
  assert.throws(
    () => prepareMeshBatchPacket([meshA, meshB], camera, 64, 64, mockWasm),
    (err) => err.message.startsWith('INCOMPATIBLE_BATCH_DEPTH:') && err.reason === 'INCOMPATIBLE_BATCH_DEPTH'
  );

  // 8. Inadmissible mesh wrapped in batch carries child reason code as prefix and reason
  const invisibleMesh = createBasicTriangleMesh();
  invisibleMesh.visible = false;
  assert.throws(
    () => prepareMeshBatchPacket([mesh, invisibleMesh], camera, 64, 64, mockWasm),
    (err) => err.message.startsWith('NOT_VISIBLE:') && err.reason === 'NOT_VISIBLE' && err.message.includes('Mesh batch admission rejected at index 1')
  );
});

test('Positive: canAdmitMesh and renderScene refuse mesh when ancestor group has visible === false (Three.js subtree culling parity)', async () => {
  const camera = createBasicCamera();

  // 1. Direct canAdmitMesh check with invisible ancestor
  const parentGroup = new THREE.Group();
  parentGroup.visible = false;
  const childMesh = createBasicTriangleMesh();
  parentGroup.add(childMesh);

  const directAdm = canAdmitMesh(childMesh, camera);
  assert.equal(directAdm.admitted, false);
  assert.equal(directAdm.reason, ADMISSION_REJECTION.NOT_VISIBLE);
  assert.equal(directAdm.code, 'NOT_VISIBLE');

  // Deeply nested ancestor invisible
  const grandParent = new THREE.Group();
  grandParent.visible = false;
  const middleGroup = new THREE.Group();
  middleGroup.visible = true;
  const deepMesh = createBasicTriangleMesh();
  middleGroup.add(deepMesh);
  grandParent.add(middleGroup);

  const deepAdm = canAdmitMesh(deepMesh, camera);
  assert.equal(deepAdm.admitted, false);
  assert.equal(deepAdm.code, 'NOT_VISIBLE');

  // 2. renderScene with mixed visibility hierarchy
  const scene = new THREE.Scene();
  const hiddenGroup = new THREE.Group();
  hiddenGroup.visible = false;
  const hiddenMesh = createBasicTriangleMesh({ color: 0xff0000 });
  hiddenGroup.add(hiddenMesh);

  const visibleMesh = createBasicTriangleMesh({ color: 0x00ff00 });
  scene.add(hiddenGroup);
  scene.add(visibleMesh);

  let capturedBatchArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedBatchArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x56, 0x49, 0x53]); // 'VIS'
    },
  };

  const mockBridgeHost = {
    executePacket: async () => ({ status: 'RENDERED' }),
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // Visible mesh admitted, hidden mesh legitimately culled without causing scene refusal
  assert.deepEqual(res.admitted, [visibleMesh.uuid]);
  assert.equal(res.refused.length, 0, 'Hidden nodes are culled and not recorded as scene refusals');

  // Batch only contains the 1 visible mesh
  assert.equal(capturedBatchArgs.vCounts.length, 1);
  assert.equal(capturedBatchArgs.cols[0], 0); // Green
  assert.equal(capturedBatchArgs.cols[1], 1);

  // 3. renderScene when only invisible meshes exist -> empty admitted set, no submit
  const hiddenScene = new THREE.Scene();
  const allHiddenGroup = new THREE.Group();
  allHiddenGroup.visible = false;
  const onlyHiddenMesh = createBasicTriangleMesh();
  allHiddenGroup.add(onlyHiddenMesh);
  hiddenScene.add(allHiddenGroup);

  let executed = false;
  const mockBridgeHostNoSubmit = {
    executePacket: async () => {
      executed = true;
      return {};
    },
  };

  const hiddenRes = await renderScene(mockBridgeHostNoSubmit, hiddenScene, camera, null, mockWasm);
  assert.deepEqual(hiddenRes.admitted, []);
  assert.equal(hiddenRes.refused.length, 0, 'Hidden meshes are culled, not recorded as refusals');
  assert.equal(hiddenRes.reason, 'EMPTY_MESH_BATCH');
  assert.equal(executed, false, 'No execution when admitted set is empty');
});

test('Positive: renderScene sorts admitted meshes ascending by renderOrder (Three.js RenderList parity)', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  // Create 4 meshes with distinct renderOrder values
  // meshA: renderOrder 10 (Red)
  const meshA = createBasicTriangleMesh({ color: 0xff0000 });
  meshA.renderOrder = 10;
  meshA.position.set(10, 0, 0);

  // meshB: renderOrder 0 (Green, default)
  const meshB = createBasicTriangleMesh({ color: 0x00ff00 });
  meshB.renderOrder = 0;
  meshB.position.set(20, 0, 0);

  // meshC: renderOrder -5 (Blue)
  const meshC = createBasicTriangleMesh({ color: 0x0000ff });
  meshC.renderOrder = -5;
  meshC.position.set(30, 0, 0);

  // meshD: renderOrder 0 (Yellow, second default)
  const meshD = createBasicTriangleMesh({ color: 0xffff00 });
  meshD.position.set(40, 0, 0); // renderOrder defaults to 0

  // Add in order: meshA, meshB, meshC, meshD
  scene.add(meshA);
  scene.add(meshB);
  scene.add(meshC);
  scene.add(meshD);

  let capturedBatchArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedBatchArgs = { flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas };
      return new Uint8Array([0x53, 0x4f, 0x52, 0x54]); // 'SORT'
    },
  };

  const mockBridgeHost = {
    executePacket: async () => ({ status: 'SORTED_OK' }),
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // Expected order:
  // -5: meshC (Blue)
  //  0: meshB (Green) - stable sort preserves order before meshD
  //  0: meshD (Yellow)
  // 10: meshA (Red)
  assert.deepEqual(res.admitted, [meshC.uuid, meshB.uuid, meshD.uuid, meshA.uuid]);
  assert.deepEqual(res.refused, []);

  // Captured colors should match the sorted order
  assert.equal(capturedBatchArgs.cols.length, 16);
  // Slot 0 (meshC, Blue: [0, 0, 1, 1])
  assert.equal(capturedBatchArgs.cols[0], 0);
  assert.equal(capturedBatchArgs.cols[1], 0);
  assert.equal(capturedBatchArgs.cols[2], 1);
  assert.equal(capturedBatchArgs.cols[3], 1);

  // Slot 1 (meshB, Green: [0, 1, 0, 1])
  assert.equal(capturedBatchArgs.cols[4], 0);
  assert.equal(capturedBatchArgs.cols[5], 1);
  assert.equal(capturedBatchArgs.cols[6], 0);
  assert.equal(capturedBatchArgs.cols[7], 1);

  // Slot 2 (meshD, Yellow: [1, 1, 0, 1])
  assert.equal(capturedBatchArgs.cols[8], 1);
  assert.equal(capturedBatchArgs.cols[9], 1);
  assert.equal(capturedBatchArgs.cols[10], 0);
  assert.equal(capturedBatchArgs.cols[11], 1);

  // Slot 3 (meshA, Red: [1, 0, 0, 1])
  assert.equal(capturedBatchArgs.cols[12], 1);
  assert.equal(capturedBatchArgs.cols[13], 0);
  assert.equal(capturedBatchArgs.cols[14], 0);
  assert.equal(capturedBatchArgs.cols[15], 1);

  // Captured model-views should also match the sorted order (X position: 30, 20, 40, 10)
  const expectedMvC = multiplyMatrices4x4(camera.matrixWorldInverse, meshC.matrixWorld);
  const expectedMvB = multiplyMatrices4x4(camera.matrixWorldInverse, meshB.matrixWorld);
  const expectedMvD = multiplyMatrices4x4(camera.matrixWorldInverse, meshD.matrixWorld);
  const expectedMvA = multiplyMatrices4x4(camera.matrixWorldInverse, meshA.matrixWorld);

  assert.deepEqual(capturedBatchArgs.mvs.slice(0, 16), expectedMvC);
  assert.deepEqual(capturedBatchArgs.mvs.slice(16, 32), expectedMvB);
  assert.deepEqual(capturedBatchArgs.mvs.slice(32, 48), expectedMvD);
  assert.deepEqual(capturedBatchArgs.mvs.slice(48, 64), expectedMvA);
});

test('Counterexample: Concrete depthWrite=false sorting discrepancy between WebGL (material.id) and WebGPU (projected z)', async () => {
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);
  camera.updateMatrixWorld();

  // Create matFar FIRST so it gets a lower material.id
  const matFar = new THREE.MeshBasicMaterial({ color: 0x0000ff, depthWrite: false, side: THREE.DoubleSide });
  // Create matNear SECOND so it gets a higher material.id
  const matNear = new THREE.MeshBasicMaterial({ color: 0xff0000, depthWrite: false, side: THREE.DoubleSide });
  assert.ok(matFar.id < matNear.id, `matFar.id (${matFar.id}) must be < matNear.id (${matNear.id})`);

  // meshNear is closer to camera (z = -2), meshFar is further (z = -10)
  const meshNear = createBasicTriangleMesh();
  meshNear.material = matNear;
  meshNear.position.set(0, 0, -2);

  const meshFar = createBasicTriangleMesh();
  meshFar.material = matFar;
  meshFar.position.set(0, 0, -10);

  const scene = new THREE.Scene();
  scene.add(meshNear);
  scene.add(meshFar);

  let capturedArgsWebGL = null;
  const mockWasmWebGL = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedArgsWebGL = { cols: Array.from(cols) };
      return new Uint8Array([0x47, 0x4c]);
    },
  };
  const mockBridgeHost = { executePacket: async () => ({ status: 'OK' }) };

  // 1. Under sourceBackend: 'webgl' (matching webgl/WebGLRenderLists.js:11)
  // WebGL groups/sorts by material.id BEFORE z.
  // Because matFar.id < matNear.id, meshFar is sorted FIRST, meshNear is sorted SECOND.
  const resWebGL = await renderScene(mockBridgeHost, scene, camera, null, mockWasmWebGL, { sourceBackend: 'webgl' });
  assert.deepEqual(resWebGL.admitted, [meshFar.uuid, meshNear.uuid]);
  // Slot 0 is meshFar (Blue: [0, 0, 1, 1]), Slot 1 is meshNear (Red: [1, 0, 0, 1])
  assert.equal(capturedArgsWebGL.cols[2], 1.0, 'Slot 0 is Blue (meshFar)');
  assert.equal(capturedArgsWebGL.cols[4], 1.0, 'Slot 1 is Red (meshNear)');
  // Concrete runtime effect: Because depthWrite=false, the last drawn object wins and is visible on screen.
  // In WebGL, meshNear (Red) is drawn second and overwrites meshFar (Blue).

  // 2. Under sourceBackend: 'webgpu' (matching common/RenderList.js:24)
  // WebGPU has no material.id sorting; it sorts by projected z directly.
  // meshNear has projected z ~1.8 < meshFar projected z ~9.8.
  // Therefore, meshNear is sorted FIRST, meshFar is sorted SECOND.
  let capturedArgsWebGPU = null;
  const mockWasmWebGPU = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, webglDepth, dt, dw, dc, canvas) => {
      capturedArgsWebGPU = { cols: Array.from(cols) };
      return new Uint8Array([0x47, 0x50, 0x55]);
    },
  };

  const resWebGPU = await renderScene(mockBridgeHost, scene, camera, null, mockWasmWebGPU, { sourceBackend: 'webgpu' });
  assert.deepEqual(resWebGPU.admitted, [meshNear.uuid, meshFar.uuid]);
  // Slot 0 is meshNear (Red), Slot 1 is meshFar (Blue)
  assert.equal(capturedArgsWebGPU.cols[0], 1.0, 'Slot 0 is Red (meshNear)');
  assert.equal(capturedArgsWebGPU.cols[6], 1.0, 'Slot 1 is Blue (meshFar)');
  // Concrete runtime effect: In WebGPU, meshFar (Blue) is drawn second and overwrites meshNear (Red).
  // Demonstrating that a renderOrder-only sort contradicts upstream backend sorting and causes opposite pixel output.
});

test('Positive: renderScene honors groupOrder over mesh renderOrder matching Three.js PainterSort', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  // Group A has higher renderOrder (5), but child meshA has lower renderOrder (0)
  const groupA = new THREE.Group();
  groupA.renderOrder = 5;
  const meshA = createBasicTriangleMesh({ color: 0xff0000 });
  meshA.renderOrder = 0;
  groupA.add(meshA);

  // Group B has lower renderOrder (1), but child meshB has higher renderOrder (10)
  const groupB = new THREE.Group();
  groupB.renderOrder = 1;
  const meshB = createBasicTriangleMesh({ color: 0x00ff00 });
  meshB.renderOrder = 10;
  groupB.add(meshB);

  scene.add(groupA);
  scene.add(groupB);

  const mockWasm = {
    f3d_build_mesh_batch_packet: () => new Uint8Array([1, 2, 3]),
  };
  const mockBridgeHost = { executePacket: async () => ({}) };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // Per Three.js painterSortStable (both WebGL and WebGPU):
  // groupOrder is compared first (a.groupOrder - b.groupOrder).
  // groupB has groupOrder 1, groupA has groupOrder 5.
  // Therefore meshB (groupOrder 1) MUST precede meshA (groupOrder 5),
  // even though meshA.renderOrder (0) < meshB.renderOrder (10).
  assert.deepEqual(res.admitted, [meshB.uuid, meshA.uuid]);
});

test('Positive: renderScene with sortObjects: false preserves scene traversal order', async () => {
  const camera = createBasicCamera();

  const mesh1 = createBasicTriangleMesh();
  mesh1.renderOrder = 100;

  const mesh2 = createBasicTriangleMesh();
  mesh2.renderOrder = -50;

  // Scene 1: traversal order [mesh1, mesh2]
  const scene1 = new THREE.Scene();
  scene1.add(mesh1);
  scene1.add(mesh2);

  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array([1]) };
  const mockBridgeHost = { executePacket: async () => ({}) };

  const res1 = await renderScene(mockBridgeHost, scene1, camera, null, mockWasm, { sortObjects: false });
  assert.deepEqual(res1.admitted, [mesh1.uuid, mesh2.uuid], 'sortObjects=false must preserve traversal order [mesh1, mesh2]');

  // Scene 2: inverted traversal order [mesh2, mesh1]
  const scene2 = new THREE.Scene();
  scene2.add(mesh2);
  scene2.add(mesh1);

  const res2 = await renderScene(mockBridgeHost, scene2, camera, null, mockWasm, { sortObjects: false });
  assert.deepEqual(res2.admitted, [mesh2.uuid, mesh1.uuid], 'sortObjects=false must preserve traversal order [mesh2, mesh1]');
});

test('Positive: renderScene respects matrixWorldAutoUpdate = false on scene and camera (renderer source boundaries)', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const mesh = createBasicTriangleMesh();
  scene.add(mesh);

  // Initialize matrices
  scene.updateMatrixWorld();
  camera.updateMatrixWorld();

  assert.equal(mesh.matrixWorld.elements[12], 0);

  // 1. Disable scene.matrixWorldAutoUpdate
  scene.matrixWorldAutoUpdate = false;
  mesh.position.set(50, 0, 0);
  mesh.updateMatrix(); // update local matrix only; matrixWorld remains 0

  let capturedMvs = null;
  const mockWasm = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs) => {
      capturedMvs = Array.from(mvs);
      return new Uint8Array([1]);
    },
  };
  const mockBridgeHost = { executePacket: async () => ({}) };

  await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  // Because scene.matrixWorldAutoUpdate === false, scene.updateMatrixWorld() was NOT called
  assert.equal(mesh.matrixWorld.elements[12], 0, 'mesh.matrixWorld must NOT be force-updated when scene.matrixWorldAutoUpdate is false');
  assert.equal(capturedMvs[12], 0, 'Emitted MV translation must remain 0');

  // 2. Disable camera.matrixWorldAutoUpdate
  camera.matrixWorldAutoUpdate = false;
  camera.position.set(0, 75, 0);
  camera.updateMatrix(); // local only

  await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
  assert.equal(camera.matrixWorld.elements[13], 0, 'camera.matrixWorld must NOT be force-updated when camera.matrixWorldAutoUpdate is false');

  // 3. Child mesh with matrixWorldAutoUpdate = false inside scene with matrixWorldAutoUpdate = true
  scene.matrixWorldAutoUpdate = true;
  mesh.matrixWorldAutoUpdate = false;
  mesh.position.set(99, 0, 0);
  mesh.updateMatrix();

  await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
  // scene.updateMatrixWorld() called without force=true honors child.matrixWorldAutoUpdate === false
  assert.equal(mesh.matrixWorld.elements[12], 0, 'Child mesh with matrixWorldAutoUpdate=false must not be forced by scene update');
});

test('Refusal & Cull: renderScene refuses visible non-mesh renderables (Line, Points, Sprite, Light) but culls them when hidden/unmatched layer', async () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array([1]) };

  // 1. Visible THREE.Line causes whole-scene refusal
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    const validMesh = createBasicTriangleMesh();
    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3));
    const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial());
    scene.add(validMesh);
    scene.add(line);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, [], 'Whole submission refused when visible Line exists');
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, line.uuid);
    assert.ok(res.refused[0].reason.includes(ADMISSION_REJECTION.UNSUPPORTED_RENDERABLE));
    assert.equal(execCount, 0, 'No packet execution on refusal');
  }

  // 2. Visible THREE.Points causes whole-scene refusal
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    const validMesh = createBasicTriangleMesh();
    const ptsGeom = new THREE.BufferGeometry();
    ptsGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3));
    const pts = new THREE.Points(ptsGeom, new THREE.PointsMaterial());
    scene.add(validMesh);
    scene.add(pts);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, pts.uuid);
    assert.equal(execCount, 0);
  }

  // 3. Visible THREE.Sprite causes whole-scene refusal
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    const validMesh = createBasicTriangleMesh();
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial());
    scene.add(validMesh);
    scene.add(sprite);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, sprite.uuid);
    assert.equal(execCount, 0);
  }

  // 4. Visible THREE.DirectionalLight causes whole-scene refusal
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    const validMesh = createBasicTriangleMesh();
    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    scene.add(validMesh);
    scene.add(light);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, light.uuid);
    assert.equal(execCount, 0);
  }

  // 5. Legitimate culls: Hidden Line and layer-culled Points do NOT block valid mesh
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return { status: 'OK' }; } };
    const scene = new THREE.Scene();
    const validMesh = createBasicTriangleMesh();

    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3));
    const hiddenLine = new THREE.Line(lineGeom, new THREE.LineBasicMaterial());
    hiddenLine.visible = false; // legitimately hidden

    const ptsGeom = new THREE.BufferGeometry();
    ptsGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3));
    const layerPts = new THREE.Points(ptsGeom, new THREE.PointsMaterial());
    layerPts.layers.set(2); // camera is on layer 0; legitimately culled by layer filter

    scene.add(validMesh);
    scene.add(hiddenLine);
    scene.add(layerPts);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, [validMesh.uuid], 'Valid mesh is admitted and rendered');
    assert.equal(res.refused.length, 0, 'Hidden/layer-culled renderables are legitimately culled without refusal');
    assert.equal(execCount, 1, 'Scene packet executed successfully');
  }
});

test('Refusal: renderScene refuses WHOLE submission when scene features (background, fog, overrideMaterial, environment) exist', async () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array([1]) };

  // 1. scene.background
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x123456);
    scene.add(createBasicTriangleMesh());

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, scene.uuid);
    assert.ok(res.refused[0].reason.includes('scene.background is not supported'));
    assert.equal(execCount, 0);
  }

  // 2. scene.fog
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xffffff, 1, 100);
    scene.add(createBasicTriangleMesh());

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, scene.uuid);
    assert.ok(res.refused[0].reason.includes('scene.fog is not supported'));
    assert.equal(execCount, 0);
  }

  // 3. scene.overrideMaterial
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    scene.overrideMaterial = new THREE.MeshBasicMaterial();
    scene.add(createBasicTriangleMesh());

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, scene.uuid);
    assert.ok(res.refused[0].reason.includes('scene.overrideMaterial is not supported'));
    assert.equal(execCount, 0);
  }

  // 4. scene.environment
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    scene.environment = new THREE.Texture();
    scene.add(createBasicTriangleMesh());

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, scene.uuid);
    assert.ok(res.refused[0].reason.includes('scene.environment is not supported'));
    assert.equal(execCount, 0);
  }
});

test('Refusal: renderScene refuses WHOLE submission when visible scene meshes have conflicting depth settings, but admits when culled', async () => {
  const camera = createBasicCamera();
  const mockWasm = { f3d_build_mesh_batch_packet: () => new Uint8Array([1]) };

  // 1. Conflicting depth settings in visible meshes -> whole-scene refusal
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return {}; } };
    const scene = new THREE.Scene();
    const meshDepthTrue = createBasicTriangleMesh({ depthTest: true });
    const meshDepthFalse = createBasicTriangleMesh({ depthTest: false });
    scene.add(meshDepthTrue);
    scene.add(meshDepthFalse);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, []);
    assert.equal(res.refused.length, 1);
    assert.equal(res.refused[0].uuid, meshDepthFalse.uuid);
    assert.ok(res.refused[0].reason.includes(ADMISSION_REJECTION.INCOMPATIBLE_BATCH_DEPTH));
    assert.equal(execCount, 0);
  }

  // 2. When the conflicting mesh is hidden -> legitimate cull, visible mesh renders
  {
    let execCount = 0;
    const mockBridgeHost = { executePacket: async () => { execCount++; return { status: 'OK' }; } };
    const scene = new THREE.Scene();
    const meshDepthTrue = createBasicTriangleMesh({ depthTest: true });
    const meshDepthFalse = createBasicTriangleMesh({ depthTest: false });
    meshDepthFalse.visible = false; // culled!
    scene.add(meshDepthTrue);
    scene.add(meshDepthFalse);

    const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
    assert.deepEqual(res.admitted, [meshDepthTrue.uuid]);
    assert.equal(res.refused.length, 0);
    assert.equal(execCount, 1);
  }
});

test('Material Side & Reflected Winding: computeAffineDeterminant and reflection parity table', () => {
  const camera = createBasicCamera();

  // Helper to test a mesh with specific side and scale
  function evaluateParity(side, scaleX, scaleY = 1, scaleZ = 1) {
    const mesh = createBasicTriangleMesh({ side });
    mesh.scale.set(scaleX, scaleY, scaleZ);
    mesh.updateMatrixWorld(true);
    const snap = extractMeshRenderData(mesh, camera, 64, 64);
    return {
      det: computeAffineDeterminant(mesh.matrixWorld),
      isReflected: snap.isReflected,
      flipSided: snap.flipSided,
      cullMode: snap.cullMode,
      frontFace: snap.frontFace,
      side: snap.side,
    };
  }

  // 1. FrontSide + normal transform (det > 0): cullMode BACK (2), frontFace CCW (0)
  const frontNormal = evaluateParity(THREE.FrontSide, 1, 1, 1);
  assert.ok(frontNormal.det > 0);
  assert.equal(frontNormal.isReflected, false);
  assert.equal(frontNormal.flipSided, false);
  assert.equal(frontNormal.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(frontNormal.frontFace, FRONT_FACE_WIRE.CCW);

  // 2. FrontSide + reflected transform (det < 0 via scale.x = -1): cullMode BACK (2), frontFace CW (1)
  const frontReflected = evaluateParity(THREE.FrontSide, -1, 1, 1);
  assert.ok(frontReflected.det < 0);
  assert.equal(frontReflected.isReflected, true);
  assert.equal(frontReflected.flipSided, true);
  assert.equal(frontReflected.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(frontReflected.frontFace, FRONT_FACE_WIRE.CW);

  // 3. BackSide + normal transform (det > 0): cullMode BACK (2), frontFace CW (1)
  const backNormal = evaluateParity(THREE.BackSide, 1, 1, 1);
  assert.ok(backNormal.det > 0);
  assert.equal(backNormal.isReflected, false);
  assert.equal(backNormal.flipSided, true);
  assert.equal(backNormal.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(backNormal.frontFace, FRONT_FACE_WIRE.CW);

  // 4. BackSide + reflected transform (det < 0 via scale.y = -1): cullMode BACK (2), frontFace CCW (0)
  const backReflected = evaluateParity(THREE.BackSide, 1, -1, 1);
  assert.ok(backReflected.det < 0);
  assert.equal(backReflected.isReflected, true);
  assert.equal(backReflected.flipSided, false);
  assert.equal(backReflected.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(backReflected.frontFace, FRONT_FACE_WIRE.CCW);

  // 5. DoubleSide + normal transform: cullMode NONE (0), frontFace CCW (0)
  const doubleNormal = evaluateParity(THREE.DoubleSide, 1, 1, 1);
  assert.equal(doubleNormal.cullMode, CULL_MODE_WIRE.NONE);
  assert.equal(doubleNormal.frontFace, FRONT_FACE_WIRE.CCW);

  // 6. DoubleSide + reflected transform: cullMode NONE (0), frontFace CW (1)
  const doubleReflected = evaluateParity(THREE.DoubleSide, -1, 1, 1);
  assert.equal(doubleReflected.cullMode, CULL_MODE_WIRE.NONE);
  assert.equal(doubleReflected.frontFace, FRONT_FACE_WIRE.CW);

  // 7. Group hierarchy reflection: Parent group with scale.x = -1 reflects child mesh
  const parentGroup = new THREE.Group();
  parentGroup.scale.set(-1, 1, 1);
  const childMesh = createBasicTriangleMesh({ side: THREE.FrontSide });
  parentGroup.add(childMesh);
  parentGroup.updateMatrixWorld(true);

  const childSnap = extractMeshRenderData(childMesh, camera, 64, 64);
  assert.equal(childSnap.isReflected, true);
  assert.equal(childSnap.flipSided, true);
  assert.equal(childSnap.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(childSnap.frontFace, FRONT_FACE_WIRE.CW);

  // 8. Double reflection cancels out (scale.x = -1, scale.y = -1 -> det > 0)
  const doubleNeg = evaluateParity(THREE.FrontSide, -1, -1, 1);
  assert.ok(doubleNeg.det > 0);
  assert.equal(doubleNeg.isReflected, false);
  assert.equal(doubleNeg.flipSided, false);
  assert.equal(doubleNeg.cullMode, CULL_MODE_WIRE.BACK);
  assert.equal(doubleNeg.frontFace, FRONT_FACE_WIRE.CCW);
});

test('Single-mesh APIs: Non-DoubleSide meshes route through f3d_build_mesh_batch_cull_packet (N=1)', () => {
  const camera = createBasicCamera();
  const frontMesh = createBasicTriangleMesh({ side: THREE.FrontSide });
  const backMesh = createBasicTriangleMesh({ side: THREE.BackSide });
  const doubleMesh = createBasicTriangleMesh({ side: THREE.DoubleSide });

  let cullCallArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_packet: (pos, vCounts, mv, proj, col, cModes, fFaces, w, h, wd, dt, dw, dc, canvas) => {
      cullCallArgs = { pos, vCounts, mv, proj, col, cModes, fFaces, w, h, wd, dt, dw, dc, canvas };
      return new Uint8Array([0xCA, 0xFE]);
    },
    f3d_build_mesh_depth_packet: () => new Uint8Array([0xDE, 0xAD]),
    f3d_build_mesh_packet: () => new Uint8Array([0xBE, 0xEF]),
  };

  // 1. prepareMeshPacket with FrontSide routes to f3d_build_mesh_batch_cull_packet
  cullCallArgs = null;
  const resFront = prepareMeshPacket(frontMesh, camera, 64, 64, mockWasm);
  assert.deepEqual(resFront.packetBytes, new Uint8Array([0xCA, 0xFE]));
  assert.equal(cullCallArgs.vCounts.length, 1);
  assert.equal(cullCallArgs.vCounts[0], 3);
  assert.deepEqual(Array.from(cullCallArgs.cModes), [CULL_MODE_WIRE.BACK]);
  assert.deepEqual(Array.from(cullCallArgs.fFaces), [FRONT_FACE_WIRE.CCW]);
  assert.equal(cullCallArgs.canvas, false);

  // 2. prepareMeshDepthPacket with BackSide routes to f3d_build_mesh_batch_cull_packet
  cullCallArgs = null;
  const resBack = prepareMeshDepthPacket(backMesh, camera, 64, 64, mockWasm);
  assert.deepEqual(resBack.packetBytes, new Uint8Array([0xCA, 0xFE]));
  assert.deepEqual(Array.from(cullCallArgs.cModes), [CULL_MODE_WIRE.BACK]);
  assert.deepEqual(Array.from(cullCallArgs.fFaces), [FRONT_FACE_WIRE.CW]);

  // 3. prepareCanvasMeshPacket with FrontSide sets canvas=true
  cullCallArgs = null;
  const resCanvasFront = prepareCanvasMeshPacket(frontMesh, camera, 64, 64, mockWasm);
  assert.deepEqual(resCanvasFront.packetBytes, new Uint8Array([0xCA, 0xFE]));
  assert.equal(cullCallArgs.canvas, true);

  // 4. prepareCanvasMeshDepthPacket with BackSide sets canvas=true
  cullCallArgs = null;
  const resCanvasBack = prepareCanvasMeshDepthPacket(backMesh, camera, 64, 64, mockWasm);
  assert.deepEqual(resCanvasBack.packetBytes, new Uint8Array([0xCA, 0xFE]));
  assert.equal(cullCallArgs.canvas, true);
  assert.deepEqual(Array.from(cullCallArgs.cModes), [CULL_MODE_WIRE.BACK]);
  assert.deepEqual(Array.from(cullCallArgs.fFaces), [FRONT_FACE_WIRE.CW]);

  // 5. DoubleSide mesh still routes through legacy depth/legacy packet builder
  cullCallArgs = null;
  const resDouble = prepareMeshPacket(doubleMesh, camera, 64, 64, mockWasm);
  assert.deepEqual(resDouble.packetBytes, new Uint8Array([0xDE, 0xAD]));
  assert.equal(cullCallArgs, null, 'DoubleSide mesh must not call cull export when legacy export exists');

  // 6. Non-DoubleSide strictly refuses when f3d_build_mesh_batch_cull_packet is missing
  const mockWasmNoCull = {
    f3d_build_mesh_depth_packet: () => new Uint8Array([1]),
    f3d_build_mesh_packet: () => new Uint8Array([1]),
  };

  assert.throws(
    () => prepareMeshPacket(frontMesh, camera, 64, 64, mockWasmNoCull),
    (err) => err.reason === 'MISSING_CULL_EXPORT' && err.message.includes('MISSING_CULL_EXPORT')
  );

  assert.throws(
    () => prepareMeshDepthPacket(backMesh, camera, 64, 64, mockWasmNoCull),
    (err) => err.reason === 'MISSING_CULL_EXPORT' && err.message.includes('MISSING_CULL_EXPORT')
  );

  assert.throws(
    () => prepareCanvasMeshPacket(frontMesh, camera, 64, 64, mockWasmNoCull),
    (err) => err.reason === 'MISSING_CULL_EXPORT' && err.message.includes('MISSING_CULL_EXPORT')
  );

  assert.throws(
    () => prepareCanvasMeshDepthPacket(backMesh, camera, 64, 64, mockWasmNoCull),
    (err) => err.reason === 'MISSING_CULL_EXPORT' && err.message.includes('MISSING_CULL_EXPORT')
  );
});

test('Multi-mesh batch: Mixed-side batch packs cullModes and frontFaces (14 arguments)', () => {
  const camera = createBasicCamera();

  const meshDouble = createBasicTriangleMesh({ color: 0x0000ff, side: THREE.DoubleSide });
  const meshFront = createBasicTriangleMesh({ color: 0x00ff00, side: THREE.FrontSide });
  const meshBack = createBasicTriangleMesh({ color: 0xff0000, side: THREE.BackSide });

  let capturedArgs = null;
  const mockWasmCull = {
    f3d_build_mesh_batch_cull_packet: (flatPos, vCounts, mvs, proj, cols, cModes, fFaces, w, h, wd, dt, dw, dc, canvas) => {
      capturedArgs = { flatPos, vCounts, mvs, proj, cols, cModes, fFaces, w, h, wd, dt, dw, dc, canvas };
      return new Uint8Array([0x14, 0x00]);
    },
  };

  const batchResult = prepareMeshBatchPacket(
    [meshDouble, meshFront, meshBack],
    camera,
    64,
    64,
    mockWasmCull,
    { target: 'offscreen' }
  );

  assert.equal(batchResult.meshCount, 3);
  assert.equal(capturedArgs.cModes.length, 3);
  assert.equal(capturedArgs.fFaces.length, 3);
  assert.deepEqual(Array.from(batchResult.cullModes), [CULL_MODE_WIRE.NONE, CULL_MODE_WIRE.BACK, CULL_MODE_WIRE.BACK]);
  assert.deepEqual(Array.from(batchResult.frontFaces), [FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CW]);
  assert.deepEqual(Array.from(capturedArgs.cModes), [0, 2, 2]);
  assert.deepEqual(Array.from(capturedArgs.fFaces), [0, 0, 1]);
  assert.equal(capturedArgs.canvas, false);

  // If Wasm only has legacy f3d_build_mesh_batch_packet, mixed-side batch throws MISSING_CULL_EXPORT
  const mockWasmLegacyOnly = {
    f3d_build_mesh_batch_packet: () => new Uint8Array([12]),
  };

  assert.throws(
    () => prepareMeshBatchPacket([meshDouble, meshFront], camera, 64, 64, mockWasmLegacyOnly),
    (err) => err.reason === 'MISSING_CULL_EXPORT' && err.message.includes('MISSING_CULL_EXPORT')
  );

  // But all-DoubleSide batch on legacy Wasm succeeds via 12-arg call
  let legacyCalled = false;
  const mockWasmLegacyOnlyDouble = {
    f3d_build_mesh_batch_packet: (flatPos, vCounts, mvs, proj, cols, w, h, wd, dt, dw, dc, canvas) => {
      legacyCalled = true;
      return new Uint8Array([12]);
    },
  };
  const doubleBatchRes = prepareMeshBatchPacket([meshDouble], camera, 64, 64, mockWasmLegacyOnlyDouble);
  assert.equal(legacyCalled, true);
  assert.equal(doubleBatchRes.meshCount, 1);
});

test('renderScene: Admits mixed sides and reflected hierarchies into single batch execution', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  // Mesh 1: Normal DoubleSide
  const mesh1 = createBasicTriangleMesh({ color: 0x0000ff, side: THREE.DoubleSide });

  // Mesh 2: Normal FrontSide
  const mesh2 = createBasicTriangleMesh({ color: 0x00ff00, side: THREE.FrontSide });

  // Mesh 3: Child of reflected group
  const reflectedGroup = new THREE.Group();
  reflectedGroup.scale.set(-1, 1, 1);
  const mesh3 = createBasicTriangleMesh({ color: 0xff0000, side: THREE.FrontSide });
  reflectedGroup.add(mesh3);

  scene.add(mesh1);
  scene.add(mesh2);
  scene.add(reflectedGroup);

  let capturedBatch = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_packet: (flatPos, vCounts, mvs, proj, cols, cModes, fFaces, w, h, wd, dt, dw, dc, canvas) => {
      capturedBatch = { cModes: Array.from(cModes), fFaces: Array.from(fFaces), meshCount: vCounts.length };
      return new Uint8Array([0xAA, 0xBB]);
    },
  };

  let executed = false;
  const mockBridgeHost = {
    executePacket: async (bytes) => {
      executed = true;
      assert.deepEqual(bytes, new Uint8Array([0xAA, 0xBB]));
      return { status: 'SCENE_OK' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  assert.equal(executed, true);
  assert.equal(res.refused.length, 0);
  assert.equal(res.admitted.length, 3);
  assert.deepEqual(res.admitted, [mesh1.uuid, mesh2.uuid, mesh3.uuid]);

  // Check cullModes and frontFaces:
  // mesh1 (DoubleSide): cull NONE (0), CCW (0)
  // mesh2 (FrontSide, det > 0): cull BACK (2), CCW (0)
  // mesh3 (FrontSide, parent reflected det < 0): cull BACK (2), CW (1)
  assert.deepEqual(capturedBatch.cModes, [CULL_MODE_WIRE.NONE, CULL_MODE_WIRE.BACK, CULL_MODE_WIRE.BACK]);
  assert.deepEqual(capturedBatch.fFaces, [FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CW]);
});

test('Mixed per-mesh depth: prepareMeshBatchPacket routes to f3d_build_mesh_batch_cull_depth_packet with 14 arguments', () => {
  const camera = createBasicCamera();

  // Mesh 1: depthTest=true, depthWrite=true, depthFunc=LessDepth (compare 2)
  const mesh1 = createBasicTriangleMesh({
    color: 0x0000ff,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });

  // Mesh 2: depthTest=false, depthWrite=true, WebGPU backend -> effective depthWrite=true, depthCompare=ALWAYS (8)
  const mesh2 = createBasicTriangleMesh({
    color: 0x00ff00,
    side: THREE.FrontSide,
    depthTest: false,
    depthWrite: true,
  });

  // Mesh 3: depthTest=true, depthWrite=false, depthFunc=GreaterDepth (compare 5)
  const mesh3 = createBasicTriangleMesh({
    color: 0xff0000,
    side: THREE.BackSide,
    depthTest: true,
    depthWrite: false,
    depthFunc: THREE.GreaterDepth,
  });

  // Mesh 4: depthTest=false, depthWrite=false -> depthCompare=ALWAYS (8)
  const mesh4 = createBasicTriangleMesh({
    color: 0xffff00,
    side: THREE.FrontSide,
    depthTest: false,
    depthWrite: false,
  });

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_packet: (
      flatPos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, w, h, wd, canvas
    ) => {
      capturedArgs = {
        vCounts: Array.from(vCounts),
        cModes: Array.from(cModes),
        fFaces: Array.from(fFaces),
        dTests: Array.from(dTests),
        dWrites: Array.from(dWrites),
        dCompares: Array.from(dCompares),
        w, h, wd, canvas,
      };
      return new Uint8Array([0xDE, 0xAD]);
    },
  };

  const res = prepareMeshBatchPacket(
    [mesh1, mesh2, mesh3, mesh4],
    camera,
    64,
    64,
    mockWasm,
    { sourceBackend: 'webgpu', target: 'offscreen' }
  );

  assert.equal(res.meshCount, 4);
  assert.deepEqual(capturedArgs.vCounts, [3, 3, 3, 3]);
  assert.deepEqual(capturedArgs.cModes, [CULL_MODE_WIRE.NONE, CULL_MODE_WIRE.BACK, CULL_MODE_WIRE.BACK, CULL_MODE_WIRE.BACK]);
  assert.deepEqual(capturedArgs.fFaces, [FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CCW, FRONT_FACE_WIRE.CW, FRONT_FACE_WIRE.CCW]);
  assert.deepEqual(capturedArgs.dTests, [1, 0, 1, 0]);
  assert.deepEqual(capturedArgs.dWrites, [1, 1, 0, 0]);
  assert.deepEqual(capturedArgs.dCompares, [
    DEPTH_WIRE_COMPARE.LESS,
    DEPTH_WIRE_COMPARE.ALWAYS,
    DEPTH_WIRE_COMPARE.GREATER,
    DEPTH_WIRE_COMPARE.ALWAYS,
  ]);
  assert.equal(capturedArgs.canvas, false);
});

test('Mixed per-mesh depth: WebGL backend resolves depthWrite=false when depthTest=false', () => {
  const camera = createBasicCamera();

  const mesh1 = createBasicTriangleMesh({
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessEqualDepth,
  });

  const mesh2 = createBasicTriangleMesh({
    depthTest: false,
    depthWrite: true, // under WebGL, depthTest=false suppresses hardware depth writes
  });

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_packet: (
      flatPos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, w, h, wd, canvas
    ) => {
      capturedArgs = {
        dTests: Array.from(dTests),
        dWrites: Array.from(dWrites),
        dCompares: Array.from(dCompares),
      };
      return new Uint8Array([0x01]);
    },
  };

  prepareMeshBatchPacket(
    [mesh1, mesh2],
    camera,
    64,
    64,
    mockWasm,
    { sourceBackend: 'webgl' }
  );

  assert.deepEqual(capturedArgs.dTests, [1, 0]);
  assert.deepEqual(capturedArgs.dWrites, [1, 0]); // Mesh 2 resolved to 0 under WebGL!
  assert.deepEqual(capturedArgs.dCompares, [DEPTH_WIRE_COMPARE.LESS_EQUAL, DEPTH_WIRE_COMPARE.ALWAYS]);
});

test('Mixed per-mesh depth: Safe refusal with INCOMPATIBLE_BATCH_DEPTH when cull_depth export is missing', () => {
  const camera = createBasicCamera();

  const mesh1 = createDepthTriangleMesh({ depthWrite: true });
  const mesh2 = createDepthTriangleMesh({ depthWrite: false });

  // Only legacy exports available
  const mockWasmLegacy = {
    f3d_build_mesh_batch_cull_packet: () => new Uint8Array(0),
    f3d_build_mesh_batch_packet: () => new Uint8Array(0),
  };

  assert.throws(
    () => prepareMeshBatchPacket([mesh1, mesh2], camera, 64, 64, mockWasmLegacy),
    (err) => err.reason === 'INCOMPATIBLE_BATCH_DEPTH' && err.message.includes('INCOMPATIBLE_BATCH_DEPTH')
  );

  // But uniform depth batch succeeds on legacy Wasm
  let cullPacketCalled = false;
  const mockWasmUniform = {
    f3d_build_mesh_batch_cull_packet: () => {
      cullPacketCalled = true;
      return new Uint8Array([0x55]);
    },
  };
  const res = prepareMeshBatchPacket([mesh1, mesh1], camera, 64, 64, mockWasmUniform);
  assert.equal(cullPacketCalled, true);
  assert.equal(res.meshCount, 2);
});

test('Mixed per-mesh depth: renderScene admits mixed depth meshes when cull_depth export is present', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const meshWriter = createBasicTriangleMesh({
    color: 0x0000ff,
    depthTest: true,
    depthWrite: true,
    depthFunc: THREE.LessDepth,
  });

  const meshOverlay = createBasicTriangleMesh({
    color: 0x00ff00,
    depthTest: false,
    depthWrite: false,
  });

  scene.add(meshWriter);
  scene.add(meshOverlay);

  let capturedArgs = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_packet: (
      flatPos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, w, h, wd, canvas
    ) => {
      capturedArgs = {
        dTests: Array.from(dTests),
        dWrites: Array.from(dWrites),
        dCompares: Array.from(dCompares),
      };
      return new Uint8Array([0x99]);
    },
  };

  let executed = false;
  const mockBridgeHost = {
    executePacket: async (bytes) => {
      executed = true;
      assert.deepEqual(bytes, new Uint8Array([0x99]));
      return { status: 'OK' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  assert.equal(executed, true);
  assert.equal(res.refused.length, 0);
  assert.equal(res.admitted.length, 2);
  assert.deepEqual(capturedArgs.dTests, [1, 0]);
  assert.deepEqual(capturedArgs.dWrites, [1, 0]);
  assert.deepEqual(capturedArgs.dCompares, [DEPTH_WIRE_COMPARE.LESS, DEPTH_WIRE_COMPARE.ALWAYS]);
});

test('Mixed per-mesh depth: renderScene safely refuses with INCOMPATIBLE_BATCH_DEPTH when cull_depth export is missing', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const meshA = createDepthTriangleMesh({ depthWrite: true });
  const meshB = createDepthTriangleMesh({ depthWrite: false });
  scene.add(meshA);
  scene.add(meshB);

  // Wasm only has legacy f3d_build_mesh_batch_cull_packet
  const mockWasmNoDepthBatch = {
    f3d_build_mesh_batch_cull_packet: () => new Uint8Array(0),
  };

  const mockBridgeHost = {
    executePacket: async () => {
      throw new Error('should not be called');
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasmNoDepthBatch);

  assert.equal(res.admitted.length, 0);
  assert.equal(res.refused.length, 1);
  assert.equal(res.reason, 'INCOMPATIBLE_BATCH_DEPTH');
});

test('Mixed per-mesh depth: renderScene refuses with AMBIGUOUS_DEPTH_PAIR when mesh has depthTest=false and depthWrite=true without sourceBackend', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();

  const meshAmbiguous = createBasicTriangleMesh({
    depthTest: false,
    depthWrite: true,
  });
  scene.add(meshAmbiguous);

  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_packet: () => new Uint8Array(0),
  };

  const mockBridgeHost = {
    executePacket: async () => {
      throw new Error('should not be called');
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);

  assert.equal(res.admitted.length, 0);
  assert.equal(res.refused.length, 1);
  assert.equal(res.reason, 'AMBIGUOUS_DEPTH_PAIR');
});

test('Single-mesh canvas routing: renderMesh admits sided N=1 with cull-depth/cull-only exports without requiring legacy canvas exports', async () => {
  const camera = createBasicCamera();
  const frontMesh = createBasicTriangleMesh({ side: THREE.FrontSide });
  const backMesh = createBasicTriangleMesh({ side: THREE.BackSide });
  const doubleMesh = createBasicTriangleMesh({ side: THREE.DoubleSide });

  const mockCanvasContext = {
    canvas: { width: 128, height: 128 },
  };

  // 1. Sided FrontSide mesh with f3d_build_mesh_batch_cull_depth_packet (no legacy canvas export)
  let capturedArgs = null;
  let executedContext = null;
  const mockBridgeHost = {
    executePacket: async (bytes, ctx) => {
      executedContext = ctx;
      return { status: 'OK' };
    },
  };

  const mockWasmCullDepth = {
    f3d_build_mesh_batch_cull_depth_packet: (
      flatPos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, w, h, wd, canvas
    ) => {
      capturedArgs = { vCounts, cModes, fFaces, w, h, canvas };
      return new Uint8Array([0x01, 0x02]);
    },
  };

  const resFrontDepth = await renderMesh(mockBridgeHost, frontMesh, camera, mockCanvasContext, mockWasmCullDepth);
  assert.equal(resFrontDepth.target, 'canvas');
  assert.equal(executedContext, mockCanvasContext);
  assert.equal(capturedArgs.canvas, true);
  assert.equal(capturedArgs.vCounts.length, 1);
  assert.equal(capturedArgs.w, 128);
  assert.equal(capturedArgs.h, 128);

  // 2. Sided BackSide mesh with f3d_build_mesh_batch_cull_packet (no legacy canvas export)
  let capturedArgsCull = null;
  const mockWasmCull = {
    f3d_build_mesh_batch_cull_packet: (
      flatPos, vCounts, mvs, proj, cols, cModes, fFaces, w, h, wd, dt, dw, dc, canvas
    ) => {
      capturedArgsCull = { vCounts, cModes, fFaces, w, h, canvas };
      return new Uint8Array([0x03, 0x04]);
    },
  };

  const resBackCull = await renderMesh(mockBridgeHost, backMesh, camera, mockCanvasContext, mockWasmCull);
  assert.equal(resBackCull.target, 'canvas');
  assert.equal(capturedArgsCull.canvas, true);

  // 3. DoubleSide mesh strictly refuses when only cull exports exist (no legacy canvas export)
  await assert.rejects(
    async () => {
      await renderMesh(mockBridgeHost, doubleMesh, camera, mockCanvasContext, mockWasmCullDepth);
    },
    {
      name: 'Error',
      message: /canvasContext provided for visible canvas rendering, but wasmModule does not export f3d_build_canvas_mesh_packet/,
    }
  );

  // 4. Sided mesh strictly refuses when neither cull export nor canvas export exists
  const mockWasmOffscreenOnly = {
    f3d_build_mesh_packet: () => new Uint8Array(0),
  };
  await assert.rejects(
    async () => {
      await renderMesh(mockBridgeHost, frontMesh, camera, mockCanvasContext, mockWasmOffscreenOnly);
    },
    {
      name: 'Error',
      message: /canvasContext provided for visible canvas rendering, but wasmModule does not export f3d_build_canvas_mesh_packet/,
    }
  );
});

test('colorWrite: prepareMeshBatchPacket admits colorWrite=false when f3d_build_mesh_batch_cull_depth_color_packet is present', () => {
  const camera = createBasicCamera();
  const occluder = createDepthTriangleMesh({ colorWrite: false, depthWrite: true });
  const visibleMesh = createDepthTriangleMesh({ colorWrite: true, depthWrite: true });

  let capturedColorWrites = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_color_packet: (
      pos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, cWrites, w, h, wd, canvas
    ) => {
      capturedColorWrites = Array.from(cWrites);
      return new Uint8Array([0xAA]);
    },
  };

  const res = prepareMeshBatchPacket([occluder, visibleMesh], camera, 64, 64, mockWasm);
  assert.deepEqual(capturedColorWrites, [0, 1]);
  assert.deepEqual(Array.from(res.colorWrites), [0, 1]);
  assert.equal(res.meshCount, 2);
});

test('colorWrite: prepareMeshBatchPacket refuses with INCOMPATIBLE_COLOR_WRITE when export is missing', () => {
  const camera = createBasicCamera();
  const occluder = createDepthTriangleMesh({ colorWrite: false });

  // Only older cull_depth export available
  const mockWasmLegacy = {
    f3d_build_mesh_batch_cull_depth_packet: () => new Uint8Array(0),
  };

  assert.throws(
    () => prepareMeshBatchPacket([occluder], camera, 64, 64, mockWasmLegacy),
    (err) => err.reason === 'INCOMPATIBLE_COLOR_WRITE' && err.message.includes('INCOMPATIBLE_COLOR_WRITE')
  );
});

test('colorWrite: renderScene admits mixed colorWrite meshes when export is present', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();
  const occluder = createDepthTriangleMesh({ colorWrite: false });
  const visible = createDepthTriangleMesh({ colorWrite: true });
  scene.add(occluder);
  scene.add(visible);

  let executed = false;
  let capturedColorWrites = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_color_packet: (
      pos, vCounts, mvs, proj, cols, cModes, fFaces, dTests, dWrites, dCompares, cWrites, w, h, wd, canvas
    ) => {
      capturedColorWrites = Array.from(cWrites);
      return new Uint8Array([0xBB]);
    },
  };
  const mockBridgeHost = {
    executePacket: async (bytes) => {
      executed = true;
      return { status: 'OK' };
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasm);
  assert.equal(executed, true);
  assert.equal(res.refused.length, 0);
  assert.equal(res.admitted.length, 2);
  assert.ok(capturedColorWrites !== null);
});

test('colorWrite: renderScene safely refuses with INCOMPATIBLE_COLOR_WRITE when export is missing', async () => {
  const scene = new THREE.Scene();
  const camera = createBasicCamera();
  const occluder = createDepthTriangleMesh({ colorWrite: false });
  scene.add(occluder);

  // Wasm lacking cull_depth_color export
  const mockWasmLegacy = {
    f3d_build_mesh_batch_cull_depth_packet: () => new Uint8Array(0),
  };
  const mockBridgeHost = {
    executePacket: async () => {
      throw new Error('should not be called');
    },
  };

  const res = await renderScene(mockBridgeHost, scene, camera, null, mockWasmLegacy);
  assert.equal(res.admitted.length, 0);
  assert.equal(res.refused.length, 1);
  assert.equal(res.reason, 'INCOMPATIBLE_COLOR_WRITE');
});

test('colorWrite: single-mesh buildSingleMeshCullPacket passes colorWrites and refuses on missing export', () => {
  const camera = createBasicCamera();
  const occluder = createDepthTriangleMesh({ colorWrite: false });
  const snap = extractMeshRenderData(occluder, camera, 64, 64);

  // Missing export throws INCOMPATIBLE_COLOR_WRITE
  const mockWasmLegacy = {
    f3d_build_mesh_batch_cull_depth_packet: () => new Uint8Array(0),
  };
  assert.throws(
    () => buildSingleMeshCullPacket(snap, 64, 64, mockWasmLegacy, false),
    (err) => err.reason === 'INCOMPATIBLE_COLOR_WRITE'
  );

  // Present export captures colorWrites = [0]
  let capturedColor = null;
  const mockWasm = {
    f3d_build_mesh_batch_cull_depth_color_packet: (
      p, v, m, pr, c, cm, ff, dt, dw, dc, cw, w, h, wd, cv
    ) => {
      capturedColor = Array.from(cw);
      return new Uint8Array([0xCC]);
    },
  };
  const pkt = buildSingleMeshCullPacket(snap, 64, 64, mockWasm, false);
  assert.deepEqual(capturedColor, [0]);
  assert.deepEqual(pkt, new Uint8Array([0xCC]));
});
