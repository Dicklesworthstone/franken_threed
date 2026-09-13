/**
 * tests/fixtures/gpu_bridge/compute_affine_test.js
 *
 * WebGPU Compute Shader Verification: AffineRows point transforms (opcodes 21/22).
 * (§6.1, §6.2, root 21362, 21411, 21413, bead f3d-05-first-frame-browser-probe).
 *
 * Tests:
 * 1. Single dispatch (f3d_build_affine_rows_compute_packet):
 *    Transforms packed vec4 points via WGSL transform_affine_point in compute storage.
 *    Validates output bit-exactly (Float32) against independent JS CPU oracle.
 * 2. Two dispatch, unaliased (f3d_build_two_dispatch_affine_compute_packet with aliased=false):
 *    Both dispatches in ONE submitted command buffer use per-use slices (offsets 0 / 256).
 *    Both output slices match independent expectations for matrix A and matrix B bit-exactly.
 * 3. Two dispatch, aliased control (aliased=true):
 *    Deliberately reuses slice A input (offset 0) in dispatch 2 (hazard counterexample).
 *    Proves detection: output B differs from matrix B expectation and matches slice A matrix A,
 * 4. Malformed-packet decoder rejection (root 21408 / 21427):
 *    Mutates a valid compute dispatch binding_type to contradict the pipeline's
 *    CreateComputePipeline bindingSpecs (BINDING_TYPE_UNIFORM vs BINDING_TYPE_STORAGE_READ).
 *    Verifies host.executePacket rejects before submission with a TypeError.
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

export const COMPUTE_AFFINE_BUFFER_ID = 701;
export const COMPUTE_INPUT_POINTS_BUFFER_ID = 702;
export const COMPUTE_OUTPUT_POINTS_BUFFER_ID = 703;
export const COMPUTE_READBACK_BUFFER_ID = 704;
export const COMPUTE_PIPELINE_ID = 300;

/**
 * Independent JS CPU oracle computing AffineRows row dot-products:
 * P' = (dot(r0, [P.xyz, 1]), dot(r1, [P.xyz, 1]), dot(r2, [P.xyz, 1]), 1.0)
 *
 * NOTE: The intermediate `Math.fround` rounding operations here reproduce
 * single-precision IEEE 754 arithmetic specifically for these test inputs.
 * This CPU oracle is NOT a general WGSL dot-product emulator across arbitrary inputs
 * (where fused multiply-add, shader compiler reordering, or varying intermediate
 * precision across GPU hardware may produce small ULP differences). Bit-exact identity
 * holds here specifically because all test inputs (matrix coefficients and point
 * coordinates) are chosen as exact dyadic fractions (denominators 1, 2, 4) and small integers
 * whose products and sums are exactly representable in Float32 without precision loss.
 */
export function cpuComputeAffinePoints(affineRows, inPoints) {
  const numPoints = inPoints.length / 4;
  const numTransforms = affineRows.length / 12;
  const expected = new Float32Array(inPoints.length);

  for (let i = 0; i < numPoints; i++) {
    const tOff = (numTransforms > 1 ? i : 0) * 12;
    const pOff = i * 4;
    const px = inPoints[pOff + 0];
    const py = inPoints[pOff + 1];
    const pz = inPoints[pOff + 2];

    const r0x = affineRows[tOff + 0], r0y = affineRows[tOff + 1], r0z = affineRows[tOff + 2], r0w = affineRows[tOff + 3];
    const r1x = affineRows[tOff + 4], r1y = affineRows[tOff + 5], r1z = affineRows[tOff + 6], r1w = affineRows[tOff + 7];
    const r2x = affineRows[tOff + 8], r2y = affineRows[tOff + 9], r2z = affineRows[tOff + 10], r2w = affineRows[tOff + 11];

    expected[pOff + 0] = Math.fround(Math.fround(r0x * px) + Math.fround(r0y * py) + Math.fround(r0z * pz) + r0w);
    expected[pOff + 1] = Math.fround(Math.fround(r1x * px) + Math.fround(r1y * py) + Math.fround(r1z * pz) + r1w);
    expected[pOff + 2] = Math.fround(Math.fround(r2x * px) + Math.fround(r2y * py) + Math.fround(r2z * pz) + r2w);
    expected[pOff + 3] = 1.0;
  }
  return expected;
}

/**
 * Bit-exact comparison of Float32Array values via 32-bit unsigned integer bit patterns.
 */
export function bitExactCompare(actual, expected) {
  if (actual.length !== expected.length) {
    return {
      match: false,
      mismatchIndex: -1,
      expected: `length ${expected.length}`,
      actual: `length ${actual.length}`,
    };
  }
  const u32Act = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const u32Exp = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);

  for (let i = 0; i < actual.length; i++) {
    if (u32Act[i] !== u32Exp[i]) {
      if (Number.isNaN(actual[i]) && Number.isNaN(expected[i])) {
        continue;
      }
      return {
        match: false,
        mismatchIndex: i,
        expected: `${expected[i]} (0x${u32Exp[i].toString(16).padStart(8, "0")})`,
        actual: `${actual[i]} (0x${u32Act[i].toString(16).padStart(8, "0")})`,
      };
    }
  }
  return { match: true };
}

function cleanupHost(host) {
  if (!host) return;
  if (host.buffers) {
    for (const b of host.buffers.values()) {
      try { b.destroy(); } catch (_) {}
    }
  }
  try {
    host.destroyDevice();
  } catch (_) {}
}

/**
 * Generates deterministic points with exact floating-point values for tail tests.
 */
export function generateDeterministicPoints(count) {
  const points = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    // Dyadic fractions / small integers for bit-exact float32 arithmetic
    points[i * 4 + 0] = (i % 7) - 3.0;
    points[i * 4 + 1] = ((i * 2) % 9) - 4.0;
    points[i * 4 + 2] = ((i * 3) % 5) + 1.0;
    points[i * 4 + 3] = 1.0;
  }
  return points;
}

/**
 * Table-driven single dispatch cases:
 * 1. Broadcast transform (1 transform for all points)
 * 2. One-per-point transforms (N transforms for N points)
 * 3. Non-64 tail workgroup (65 points exercising workgroup boundary: 64 in wg0, 1 in wg1, 63 tail threads)
 */
export const SINGLE_DISPATCH_CASES = [
  {
    key: "compute_affine_broadcast",
    name: "Broadcast Transform",
    description: "1 transform broadcast to all points (with shear/axis-mixing)",
    affine: new Float32Array([
      1.0, 0.5, -0.25, 1.5,
      -0.5, 2.0, 0.25, -2.0,
      0.25, -0.5, 1.0, 4.0,
    ]),
    points: new Float32Array([
      1.0, 2.0, 4.0, 1.0,
      -2.0, 0.0, 6.0, 1.0,
      0.5, -1.0, 2.0, 1.0,
      3.0, 1.0, -4.0, 1.0,
    ]),
  },
  {
    key: "compute_affine_one_per_point",
    name: "One-Per-Point Transforms",
    description: "N transforms for N points",
    affine: new Float32Array([
      // T0: Scale 2, Trans x=1
      2.0, 0.0, 0.0, 1.0,
      0.0, 2.0, 0.0, 0.0,
      0.0, 0.0, 2.0, 0.0,
      // T1: Scale 0.5, Trans y=5
      0.5, 0.0, 0.0, 0.0,
      0.0, 0.5, 0.0, 5.0,
      0.0, 0.0, 0.5, 0.0,
      // T2: Trans z=-3
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, -3.0,
    ]),
    points: new Float32Array([
      2.0, 2.0, 2.0, 1.0,
      4.0, 2.0, 6.0, 1.0,
      1.0, 3.0, 5.0, 1.0,
    ]),
  },
  {
    key: "compute_affine_non_64_tail",
    name: "Non-64 Tail Workgroup",
    description: "65 points exercising workgroup tail (64 in wg0, 1 in wg1, 63 inactive threads)",
    affine: new Float32Array([
      1.5, 0.0, 0.0, 0.5,
      0.0, 1.5, 0.0, -0.5,
      0.0, 0.0, 1.5, 1.0,
    ]),
    points: generateDeterministicPoints(65),
  },
];

/**
 * Test Case 1: Runs all table-driven single dispatch compute transformation cases.
 */
export async function testComputeAffineSingleDispatch(wasmExports, host) {
  const buildFn =
    wasmExports.f3d_build_affine_rows_compute_packet ||
    wasmExports.gpu_bridge_build_affine_rows_compute_packet;
  if (typeof buildFn !== "function") {
    throw new Error("Missing required export f3d_build_affine_rows_compute_packet on wasmExports");
  }

  const results = {};
  for (const testCase of SINGLE_DISPATCH_CASES) {
    const { key, name, description, affine, points } = testCase;
    const expected = cpuComputeAffinePoints(affine, points);
    const packetBytes = buildFn(affine, points);
    await host.executePacket(packetBytes);

    const readbackId = wasmExports.COMPUTE_READBACK_BUFFER_ID ?? COMPUTE_READBACK_BUFFER_ID;
    const readbackBytes = await host.readbackBuffer(readbackId, points.length * 4);
    const observed = new Float32Array(readbackBytes.buffer, readbackBytes.byteOffset, points.length);

    const cmp = bitExactCompare(observed, expected);
    if (!cmp.match) {
      results[key] = {
        status: "FAIL",
        name,
        error: `${name} mismatch at float[${cmp.mismatchIndex}]: expected ${cmp.expected}, observed ${cmp.actual}`,
        detail: `${name} mismatch at float[${cmp.mismatchIndex}]: expected ${cmp.expected}, observed ${cmp.actual}`,
        points_count: points.length / 4,
        expected: Array.from(expected.slice(0, 16)),
        observed: Array.from(observed.slice(0, 16)),
        implementation_owner: "new-backend",
      };
      return {
        status: "FAIL",
        subcases: results,
        error: results[key].error,
        detail: results[key].detail,
        implementation_owner: "new-backend",
      };
    }

    results[key] = {
      status: "PASS",
      name,
      detail: `${name} (${points.length / 4} points, ${description}): bit-exactly matches CPU oracle (owner: new-backend)`,
      points_count: points.length / 4,
      expected: Array.from(expected.slice(0, 16)),
      observed: Array.from(observed.slice(0, 16)),
      implementation_owner: "new-backend",
    };
  }

  return {
    status: "PASS",
    subcases: results,
    detail: `Single dispatch table (${SINGLE_DISPATCH_CASES.length} cases: broadcast, one-per-point, non-64 tail 65-pts): all bit-exact matches (owner: new-backend)`,
    implementation_owner: "new-backend",
  };
}

/**
 * Test Case 2: Two dispatch, unaliased dispatches in ONE submission.
 */
export async function testComputeAffineTwoDispatchUnaliased(wasmExports, host) {
  const buildTwoFn =
    wasmExports.f3d_build_two_dispatch_affine_compute_packet ||
    wasmExports.gpu_bridge_build_two_dispatch_affine_compute_packet;
  if (typeof buildTwoFn !== "function") {
    throw new Error("Missing required export f3d_build_two_dispatch_affine_compute_packet on wasmExports");
  }

  // Matrix A: Scale (2, 2, 2), Translation (1, 0, 0)
  const matrixA = new Float32Array([
    2.0, 0.0, 0.0, 1.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 0.0,
  ]);
  // Matrix B: Scale (0.5, 0.5, 0.5), Translation (0, 5, 0)
  const matrixB = new Float32Array([
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 5.0,
    0.0, 0.0, 0.5, 0.0,
  ]);
  // 2 packed vec4 points (8 floats)
  const points = new Float32Array([
    2.0, 2.0, 2.0, 1.0,
    1.0, -4.0, 6.0, 1.0,
  ]);

  const expectedA = cpuComputeAffinePoints(matrixA, points);
  const expectedB = cpuComputeAffinePoints(matrixB, points);

  const packetBytes = buildTwoFn(matrixA, matrixB, points, false);
  await host.executePacket(packetBytes);

  const pointsByteLen = points.length * 4;
  const outputSliceStride = Math.floor((pointsByteLen + 255) / 256) * 256;
  const totalOutputSize = outputSliceStride * 2;

  const readbackId = wasmExports.COMPUTE_READBACK_BUFFER_ID ?? COMPUTE_READBACK_BUFFER_ID;
  const readbackBytes = await host.readbackBuffer(readbackId, totalOutputSize);

  const observedA = new Float32Array(readbackBytes.buffer, readbackBytes.byteOffset, points.length);
  const observedB = new Float32Array(readbackBytes.buffer, readbackBytes.byteOffset + outputSliceStride, points.length);

  const cmpA = bitExactCompare(observedA, expectedA);
  if (!cmpA.match) {
    return {
      status: "FAIL",
      error: `Two dispatch unaliased slice A mismatch at float[${cmpA.mismatchIndex}]: expected ${cmpA.expected}, observed ${cmpA.actual}`,
      detail: `Two dispatch unaliased slice A mismatch at float[${cmpA.mismatchIndex}]: expected ${cmpA.expected}, observed ${cmpA.actual}`,
      output_a: { expected: Array.from(expectedA), observed: Array.from(observedA) },
      output_b: { expected: Array.from(expectedB), observed: Array.from(observedB) },
      implementation_owner: "new-backend",
    };
  }

  const cmpB = bitExactCompare(observedB, expectedB);
  if (!cmpB.match) {
    return {
      status: "FAIL",
      error: `Two dispatch unaliased slice B mismatch at float[${cmpB.mismatchIndex}]: expected ${cmpB.expected}, observed ${cmpB.actual}`,
      detail: `Two dispatch unaliased slice B mismatch at float[${cmpB.mismatchIndex}]: expected ${cmpB.expected}, observed ${cmpB.actual}`,
      output_a: { expected: Array.from(expectedA), observed: Array.from(observedA) },
      output_b: { expected: Array.from(expectedB), observed: Array.from(observedB) },
      implementation_owner: "new-backend",
    };
  }

  return {
    status: "PASS",
    detail: `Two dispatch unaliased: slice A ([${Array.from(observedA.slice(0, 4)).join(", ")}]) and slice B ([${Array.from(observedB.slice(0, 4)).join(", ")}]) match independent expectations bit-exactly (owner: new-backend)`,
    output_a: { expected: Array.from(expectedA), observed: Array.from(observedA) },
    output_b: { expected: Array.from(expectedB), observed: Array.from(observedB) },
    implementation_owner: "new-backend",
  };
}

/**
 * Test Case 3: Two dispatch, aliased control (mutation / input aliasing counterexample).
 */
export async function testComputeAffineTwoDispatchAliased(wasmExports, host) {
  const buildTwoFn =
    wasmExports.f3d_build_two_dispatch_affine_compute_packet ||
    wasmExports.gpu_bridge_build_two_dispatch_affine_compute_packet;
  if (typeof buildTwoFn !== "function") {
    throw new Error("Missing required export f3d_build_two_dispatch_affine_compute_packet on wasmExports");
  }

  const matrixA = new Float32Array([
    2.0, 0.0, 0.0, 1.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 0.0,
  ]);
  const matrixB = new Float32Array([
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 5.0,
    0.0, 0.0, 0.5, 0.0,
  ]);
  const points = new Float32Array([
    2.0, 2.0, 2.0, 1.0,
    1.0, -4.0, 6.0, 1.0,
  ]);

  const expectedA = cpuComputeAffinePoints(matrixA, points);
  const expectedB = cpuComputeAffinePoints(matrixB, points);

  // aliased=true deliberately reuses Slice A (offset 0) in dispatch 2
  const packetBytes = buildTwoFn(matrixA, matrixB, points, true);
  await host.executePacket(packetBytes);

  const pointsByteLen = points.length * 4;
  const outputSliceStride = Math.floor((pointsByteLen + 255) / 256) * 256;
  const totalOutputSize = outputSliceStride * 2;

  const readbackId = wasmExports.COMPUTE_READBACK_BUFFER_ID ?? COMPUTE_READBACK_BUFFER_ID;
  const readbackBytes = await host.readbackBuffer(readbackId, totalOutputSize);

  const observedA = new Float32Array(readbackBytes.buffer, readbackBytes.byteOffset, points.length);
  const observedB = new Float32Array(readbackBytes.buffer, readbackBytes.byteOffset + outputSliceStride, points.length);

  // 1. Output A must match matrix A
  const cmpA = bitExactCompare(observedA, expectedA);
  if (!cmpA.match) {
    return {
      status: "FAIL",
      error: `Aliased control dispatch 1 slice A mismatch: expected ${cmpA.expected}, observed ${cmpA.actual}`,
      detail: `Aliased control dispatch 1 slice A mismatch: expected ${cmpA.expected}, observed ${cmpA.actual}`,
      output_a: Array.from(observedA),
      output_b: Array.from(observedB),
      implementation_owner: "new-backend",
    };
  }

  // 2. Output B must DIFFER from matrix B expectation (proves aliasing hazard detection)
  const cmpBDiff = bitExactCompare(observedB, expectedB);
  if (cmpBDiff.match) {
    return {
      status: "FAIL",
      error: `Aliased control failed to detect hazard: output B unexpectedly matched unaliased matrix B expectations`,
      detail: `Aliased control failed to detect hazard: output B unexpectedly matched unaliased matrix B expectations`,
      output_a: Array.from(observedA),
      output_b: Array.from(observedB),
      implementation_owner: "new-backend",
    };
  }

  // 3. Output B must match matrix A expectation (documented aliasing behavior: slice A reused)
  const cmpBMatchA = bitExactCompare(observedB, expectedA);
  if (!cmpBMatchA.match) {
    return {
      status: "FAIL",
      error: `Aliased control output B did not match reused slice A matrix A: expected ${cmpBMatchA.expected}, observed ${cmpBMatchA.actual}`,
      detail: `Aliased control output B did not match reused slice A matrix A: expected ${cmpBMatchA.expected}, observed ${cmpBMatchA.actual}`,
      output_a: Array.from(observedA),
      output_b: Array.from(observedB),
      implementation_owner: "new-backend",
    };
  }

  // 4. Non-trivial check: outputs must be non-zero and non-trivial
  const nonTrivial = observedB.some(val => val !== 0 && !Number.isNaN(val));
  if (!nonTrivial) {
    return {
      status: "FAIL",
      error: `Aliased control output B is all zeros or NaN; vacuous pass rejected`,
      detail: `Aliased control output B is all zeros or NaN; vacuous pass rejected`,
      output_a: Array.from(observedA),
      output_b: Array.from(observedB),
      implementation_owner: "new-backend",
    };
  }

  return {
    status: "PASS",
    detail: `Two dispatch aliased control: dispatch 2 reused slice A offset 0; output B differed from matrix B and matched slice A ([${Array.from(observedB.slice(0, 4)).join(", ")}]) (non-trivial, hazard detected; owner: new-backend)`,
    output_a: Array.from(observedA),
    output_b: Array.from(observedB),
    expected_matrix_a: Array.from(expectedA),
    expected_matrix_b: Array.from(expectedB),
    detected_difference: true,
    matched_reused_slice_a: true,
    non_trivial_render: true,
    implementation_owner: "new-backend",
  };
}

/**
 * Locates the byte offset of the `binding_type` field for the specified binding index
 * within the first DISPATCH_COMPUTE command in an encoded packet.
 */
export function findDispatchComputeBindingTypeOffset(packetBytes, targetBindingIndex = 0) {
  const dataView = new DataView(packetBytes.buffer, packetBytes.byteOffset, packetBytes.byteLength);
  const commandCount = dataView.getUint32(8, true);
  let cursor = 16;
  for (let i = 0; i < commandCount; i++) {
    const opcode = dataView.getUint16(cursor, true);
    cursor += 2;
    if (opcode === 1) { // OPCODE_CREATE_BUFFER
      cursor += 12;
    } else if (opcode === 2) { // OPCODE_WRITE_BUFFER
      cursor += 16;
    } else if (opcode === 20) { // OPCODE_COPY_BUFFER_TO_BUFFER
      cursor += 40;
    } else if (opcode === 21) { // OPCODE_CREATE_COMPUTE_PIPELINE
      const bindingCount = dataView.getUint32(cursor + 22, true);
      cursor += 30 + bindingCount * 16;
    } else if (opcode === 22) { // OPCODE_DISPATCH_COMPUTE
      const bindingCount = dataView.getUint32(cursor + 18, true);
      const bindingsStart = cursor + 22;
      for (let b = 0; b < bindingCount; b++) {
        const bStart = bindingsStart + b * 48;
        const bIndex = dataView.getUint32(bStart, true);
        if (bIndex === targetBindingIndex) {
          // binding_type is u32 at byte offset 24 within the 48-byte binding record
          return bStart + 24;
        }
      }
      cursor += 22 + bindingCount * 48;
    } else {
      throw new Error(`findDispatchComputeBindingTypeOffset: unhandled opcode ${opcode} at command ${i}`);
    }
  }
  return -1;
}

/**
 * Test Case 4: Malformed-packet decoder rejection (root 21408 / 21427).
 *
 * Verifies that the host decoder (bridge_runtime.js) enforces pipeline bindingSpecs:
 * when a DISPATCH_COMPUTE command specifies a binding_type that contradicts
 * the pipeline's CreateComputePipeline layout (e.g. BINDING_TYPE_UNIFORM = 0
 * instead of BINDING_TYPE_STORAGE_READ = 1), executePacket synchronously rejects
 * the packet before submission with a TypeError.
 */
export async function testComputeAffineMalformedBindingSpecs(wasmExports, host) {
  const buildFn =
    wasmExports.f3d_build_affine_rows_compute_packet ||
    wasmExports.gpu_bridge_build_affine_rows_compute_packet;
  if (typeof buildFn !== "function") {
    throw new Error("Missing required export f3d_build_affine_rows_compute_packet on wasmExports");
  }

  const affine = SINGLE_DISPATCH_CASES[0].affine;
  const points = SINGLE_DISPATCH_CASES[0].points;
  const validPacket = buildFn(affine, points);

  const malformed = validPacket.slice();
  const offset = findDispatchComputeBindingTypeOffset(malformed, 0);
  if (offset < 0) {
    return {
      status: "FAIL",
      error: "Failed to locate DispatchCompute binding 0 binding_type offset in compute packet",
      detail: "Failed to locate DispatchCompute binding 0 binding_type offset in compute packet",
      implementation_owner: "new-backend",
    };
  }

  const view = new DataView(malformed.buffer, malformed.byteOffset, malformed.byteLength);
  const originalType = view.getUint32(offset, true);
  // Contradict pipeline spec: pipeline expects BINDING_TYPE_STORAGE_READ (1); mutate to BINDING_TYPE_UNIFORM (0)
  const contradictoryType = originalType === 1 ? 0 : 1;
  view.setUint32(offset, contradictoryType, true);

  let submittedObserved = false;
  let caughtError = null;
  try {
    await host.executePacket(malformed, null, () => {
      submittedObserved = true;
    });
  } catch (err) {
    caughtError = err;
  }

  if (!caughtError) {
    return {
      status: "FAIL",
      error: "Malformed packet with contradictory binding_type was accepted without error",
      detail: "Malformed packet with contradictory binding_type was accepted without error",
      implementation_owner: "new-backend",
    };
  }

  if (submittedObserved) {
    return {
      status: "FAIL",
      error: "Malformed packet notified submission despite contradicting pipeline bindingSpecs",
      detail: "Malformed packet notified submission despite contradicting pipeline bindingSpecs",
      implementation_owner: "new-backend",
    };
  }

  const isExpectedTypeMismatch =
    caughtError instanceof TypeError &&
    /binding index 0 type mismatch/.test(caughtError.message);

  if (!isExpectedTypeMismatch) {
    return {
      status: "FAIL",
      error: `Expected TypeError for bindingSpecs mismatch, got: ${caughtError.name}: ${caughtError.message}`,
      detail: `Expected TypeError for bindingSpecs mismatch, got: ${caughtError.name}: ${caughtError.message}`,
      implementation_owner: "new-backend",
    };
  }

  return {
    status: "PASS",
    detail: `Malformed bindingSpecs rejected: ${caughtError.message} (owner: new-backend)`,
    caught_error: caughtError.message,
    original_type: originalType,
    contradictory_type: contradictoryType,
    submission_prevented: true,
    implementation_owner: "new-backend",
  };
}

/**
 * Main entry point: runs all four compute affine tests.
 */
export async function testComputeAffine(wasmExports, optionalHost = null) {
  let host = optionalHost;
  let ownedHost = false;
  if (!host || !host.device) {
    host = new WebGpuBridgeHost();
    await host.negotiateAndCreateDevice({ requiredFeatures: [] });
    ownedHost = true;
  }

  const host_environment = (typeof navigator !== "undefined" && navigator.userAgent) ? navigator.userAgent : "Node/Unknown";
  const cases = {};

  try {
    try {
      const singleResults = await testComputeAffineSingleDispatch(wasmExports, host);
      if (singleResults.subcases) {
        Object.assign(cases, singleResults.subcases);
      } else {
        cases.compute_affine_single_dispatch = singleResults;
      }
    } catch (err) {
      cases.compute_affine_single_dispatch = {
        status: "FAIL",
        error: err.message || String(err),
        detail: err.message || String(err),
        implementation_owner: "new-backend",
      };
    }

    try {
      cases.compute_affine_two_dispatch_unaliased = await testComputeAffineTwoDispatchUnaliased(wasmExports, host);
    } catch (err) {
      cases.compute_affine_two_dispatch_unaliased = {
        status: "FAIL",
        error: err.message || String(err),
        detail: err.message || String(err),
        implementation_owner: "new-backend",
      };
    }

    try {
      cases.compute_affine_two_dispatch_aliased_control = await testComputeAffineTwoDispatchAliased(wasmExports, host);
    } catch (err) {
      cases.compute_affine_two_dispatch_aliased_control = {
        status: "FAIL",
        error: err.message || String(err),
        detail: err.message || String(err),
        implementation_owner: "new-backend",
      };
    }

    try {
      cases.compute_affine_malformed_binding_specs = await testComputeAffineMalformedBindingSpecs(wasmExports, host);
    } catch (err) {
      cases.compute_affine_malformed_binding_specs = {
        status: "FAIL",
        error: err.message || String(err),
        detail: err.message || String(err),
        implementation_owner: "new-backend",
      };
    }

    const allPass = Object.values(cases).every(c => c.status === "PASS");

    return {
      status: allPass ? "PASS" : "FAIL",
      host_environment,
      cases,
      detail: `Compute Affine: ${Object.values(cases).filter(c => c.status === "PASS").length}/${Object.keys(cases).length} checks PASS (broadcast, one-per-point, non-64 tail, two-dispatch unaliased, two-dispatch aliased control, malformed bindingSpecs rejection; owner: new-backend)`,
      implementation_owner: "new-backend",
    };
  } finally {
    if (ownedHost) {
      cleanupHost(host);
    }
  }
}
