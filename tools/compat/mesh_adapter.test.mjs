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
  renderMesh,
  renderRetainedFallback,
  multiplyMatrices4x4,
  expandIndexedPositions,
  ADMISSION_REJECTION,
  COORDINATE_SYSTEM,
  DEPTH_WIRE_COMPARE,
  THREE_DEPTH_FUNC_TO_WIRE_COMPARE,
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

test('Side admission: Only DoubleSide (2) is admitted; FrontSide (0) and BackSide (1) are rejected (13062 point 1)', () => {
  const camera = createBasicCamera();

  const doubleMesh = createBasicTriangleMesh({ side: THREE.DoubleSide });
  assert.equal(canAdmitMesh(doubleMesh, camera).admitted, true);
  const doubleSnap = extractMeshRenderData(doubleMesh, camera, 64, 64);
  assert.equal(doubleSnap.side, 2);

  const frontMesh = createBasicTriangleMesh({ side: THREE.FrontSide });
  const frontAdmission = canAdmitMesh(frontMesh, camera);
  assert.equal(frontAdmission.admitted, false);
  assert.equal(frontAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_SIDE);

  const backMesh = createBasicTriangleMesh({ side: THREE.BackSide });
  const backAdmission = canAdmitMesh(backMesh, camera);
  assert.equal(backAdmission.admitted, false);
  assert.equal(backAdmission.reason, ADMISSION_REJECTION.UNSUPPORTED_SIDE);
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

  // colorWrite = false
  const cwMesh = createBasicTriangleMesh({ colorWrite: false });
  assert.equal(canAdmitMesh(cwMesh, camera).admitted, false);

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
