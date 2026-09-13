// tmt.5 workload (b) prototype: packed 48-byte AffineRows layout upload/copy/readback proof of 4,000 records (§6.4, §16.7, §21).
// Bit-exact verification of Wasm packer against input DataView, preserving -0.0 and quiet NaN (0x7fc01337).
// Executes two sequential uploads (queue drain between them; no concurrency or completed tmt.5 claim).
// Buffers are COPY_SRC|COPY_DST; this is a packed-layout upload/readback proof only.

import { createDirectPersistentPackedBufferRunner } from "./direct_reference.js";

export const WORKLOAD_B_MEASURE_ROUNDS = 5;
export const WORKLOAD_B_MEASURE_WARMUP = 16;
export const WORKLOAD_B_MEASURE_MEASURED = 240;
export const WORKLOAD_B_RECORD_COUNT = 4000;
export const WORKLOAD_B_EXPECTED_BYTES = 192000; // 4000 records * 48 bytes

function generateDistinctAffineRows(recordCount, uploadIndex) {
  const buffer = new ArrayBuffer(recordCount * 12 * 4);
  const floats = new Float32Array(buffer);
  const uint32s = new Uint32Array(buffer);
  const delta = uploadIndex * 0.125;

  for (let i = 0; i < recordCount; i++) {
    const base = i * 12;
    floats[base + 0] = 1.0 + (i * 0.0001) + delta;
    floats[base + 1] = 0.01 * ((i % 7) + 1);
    floats[base + 2] = 0.02 * ((i % 11) + 1);
    floats[base + 3] = (i * 0.25) + 1.0 + delta;
    floats[base + 4] = 0.03 * ((i % 5) + 1);
    floats[base + 5] = 1.0 + (i * 0.0002) + delta;
    floats[base + 6] = 0.04 * ((i % 13) + 1);
    floats[base + 7] = (i * 0.5) - 2.0 + delta;
    floats[base + 8] = 0.05 * ((i % 9) + 1);
    floats[base + 9] = 0.06 * ((i % 17) + 1);
    floats[base + 10] = 1.0 + (i * 0.0003) + delta;
    floats[base + 11] = (i * 0.75) + 3.0 + delta;
  }

  // Inject -0.0 (0x80000000) and quiet NaN with payload 0x1337 (0x7fc01337) via Uint32Array view
  // to avoid JS Number NaN canonicalization across the JS->Wasm boundary.
  uint32s[1] = 0x80000000;
  uint32s[2] = 0x7fc01337;
  return { buffer, floats };
}

function buildExpectedBytes(buffer, floatCount) {
  const expected = new Uint8Array(floatCount * 4);
  const dvSrc = new DataView(buffer);
  const dvDst = new DataView(expected.buffer);
  for (let i = 0; i < floatCount; i++) {
    dvDst.setUint32(i * 4, dvSrc.getUint32(i * 4, true), true);
  }
  return expected;
}

function assertSpecialBits(bytes, label, upload) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const negZero = dv.getUint32(4, true);
  if (negZero !== 0x80000000) throw new Error(`Upload ${upload}: ${label} -0.0 mismatch: 0x${negZero.toString(16)}`);
  const quietNan = dv.getUint32(8, true);
  if (quietNan !== 0x7fc01337) throw new Error(`Upload ${upload}: ${label} quiet NaN mismatch: 0x${quietNan.toString(16)}`);
}

function assertBytesEqual(actual, expected, label, upload) {
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(`Upload ${upload}: ${label} byte mismatch at ${i}: got ${actual[i]}, expected ${expected[i]}`);
    }
  }
}

async function drainAndCleanup(host, srcId, dstId) {
  try {
    if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
  } finally {
    for (const id of [srcId, dstId]) {
      if (host.buffers?.has(id)) {
        host.buffers.get(id).destroy();
        host.buffers.delete(id);
      }
      host.bufferEpochs?.delete(id);
    }
  }
}

export async function testWorkloadBStorageCopy(host, wasmExports) {
  const packFn = wasmExports?.f3d_pack_affine_rows_storage_bytes;
  if (typeof packFn !== "function") {
    throw new Error("Missing required canonical export: f3d_pack_affine_rows_storage_bytes");
  }
  const buildCopyPacket = wasmExports?.f3d_build_buffer_copy_packet;
  if (typeof buildCopyPacket !== "function") {
    throw new Error("Missing required canonical export: f3d_build_buffer_copy_packet");
  }
  if (!host || !host.device) throw new Error("WebGpuBridgeHost device not initialized");

  const recordCount = 4000;
  const floatCount = recordCount * 12; // 48,000 f32 values
  const expectedBytes = recordCount * 48; // 192,000 bytes
  const srcBufferId = 610;
  const dstBufferId = 611;
  const uploadReports = [];

  // Two sequential uploads with drain and buffer destruction between them.
  // Claims NO concurrency and NO completed tmt.5 workload.
  for (let upload = 0; upload < 2; upload++) {
    const { buffer, floats } = generateDistinctAffineRows(recordCount, upload);
    const expected = buildExpectedBytes(buffer, floatCount);

    const packed = packFn(floats);
    if (!(packed instanceof Uint8Array) || packed.byteLength !== expectedBytes) {
      throw new Error(`Upload ${upload}: packed length mismatch, got ${packed?.byteLength}, expected ${expectedBytes}`);
    }

    // 1. Verify Wasm packer output matches expected bytes independently built from input floats
    assertBytesEqual(packed, expected, "packed vs expected", upload);
    // 2. Assert exact bit patterns for -0.0 and quiet NaN survived packing
    assertSpecialBits(packed, "packed", upload);

    // 3. Build opcode 20 buffer-to-buffer copy packet and execute on real device
    const packet = buildCopyPacket(packed, 0, 0, expectedBytes);
    if (!(packet instanceof Uint8Array) || packet.byteLength === 0) {
      throw new Error(`Upload ${upload}: f3d_build_buffer_copy_packet returned empty or invalid packet`);
    }

    try {
      await host.executePacket(packet);
      const readback = await host.readbackBuffer(dstBufferId, expectedBytes);
      if (readback.byteLength !== expectedBytes) {
        throw new Error(`Upload ${upload}: readback length mismatch: got ${readback.byteLength}, expected ${expectedBytes}`);
      }

      // 4. Assert readback matches packed bytes byte-for-byte
      assertBytesEqual(readback, packed, "readback vs packed", upload);
      // 5. Assert exact bit patterns survived readback without canonicalization
      assertSpecialBits(readback, "readback", upload);

      uploadReports.push(`upload ${upload}: ${expectedBytes} bytes byte-exact, -0.0/NaN bits verified`);
    } finally {
      await drainAndCleanup(host, srcBufferId, dstBufferId);
    }
  }

  return `Workload (b) packed-layout upload/readback proof: 4000 records (192000 bytes/upload) verified byte-exact with preserved -0.0 and quiet NaN across two sequential uploads [${uploadReports.join("; ")}] (no concurrency claim, no completed tmt.5 workload)`;
}

function calculateMedian(arr) {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return (s.length % 2 !== 0) ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function calculateP95(arr) {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.floor(s.length * 0.95));
  return s[idx];
}

/**
 * Creates an execution wrapper for one of the four workload (b) variants.
 * Handles persistent buffer allocation, per-frame submission, readback, and destruction.
 */
function createVariant(key, host, wasmExports, recordCount = WORKLOAD_B_RECORD_COUNT, expectedBytes = WORKLOAD_B_EXPECTED_BYTES) {
  switch (key) {
    case "workload_b_bulk_packet": {
      return {
        name: "workload_b_bulk_packet",
        implementation_owner: "Rust/Wasm packer and packet encoder; JavaScript WebGPU adapter",
        init: async () => {
          const initFn = wasmExports?.f3d_build_affine_rows_storage_upload_init_packet;
          if (typeof initFn !== "function") {
            throw new Error("Missing required canonical export: f3d_build_affine_rows_storage_upload_init_packet");
          }
          const frameFn = wasmExports?.f3d_build_affine_rows_storage_upload_frame_packet;
          if (typeof frameFn !== "function") {
            throw new Error("Missing required canonical export: f3d_build_affine_rows_storage_upload_frame_packet");
          }
          const initPacket = initFn(recordCount);
          if (!(initPacket instanceof Uint8Array) || initPacket.byteLength === 0) {
            throw new Error("f3d_build_affine_rows_storage_upload_init_packet returned empty or invalid packet");
          }
          await host.executePacket(initPacket);
          for (const id of [610, 611, 612]) {
            if (!host.buffers?.has(id)) {
              throw new Error(`Bulk init packet did not create expected buffer ${id}`);
            }
          }
        },
        submitFrame: (floats, slot) => {
          const packet = wasmExports.f3d_build_affine_rows_storage_upload_frame_packet(floats, recordCount, slot);
          if (!(packet instanceof Uint8Array) || packet.byteLength === 0) {
            throw new Error("f3d_build_affine_rows_storage_upload_frame_packet returned empty or invalid packet");
          }
          const packetPromise = host.executePacket(packet);
          const queuePromise = host.device.queue.onSubmittedWorkDone();
          return Promise.all([packetPromise, queuePromise]);
        },
        readback: async (slot) => {
          return host.readbackBuffer(slot, expectedBytes);
        },
        destroy: async () => {
          try {
            if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
          } finally {
            for (const id of [610, 611, 612]) {
              if (host.buffers?.has(id)) {
                host.buffers.get(id).destroy();
                host.buffers.delete(id);
              }
              host.bufferEpochs?.delete(id);
            }
          }
        },
      };
    }

    case "workload_b_wasm_callbacks": {
      let cbBuffers = null;
      let prevWriteBuffer = undefined;
      let prevCopyBuffer = undefined;
      const globalScope = typeof window !== "undefined" ? window : globalThis;
      globalScope.f3dHost = globalScope.f3dHost || {};
      const hostObj = globalScope.f3dHost;

      return {
        name: "workload_b_wasm_callbacks",
        implementation_owner: "Rust/Wasm callback loop; JavaScript WebGPU adapter",
        init: async () => {
          if (typeof wasmExports?.f3d_bridge_callback_storage_upload_frame !== "function") {
            throw new Error("Missing required canonical export: f3d_bridge_callback_storage_upload_frame");
          }
          const src = host.device.createBuffer({
            label: "f3d-callback-src-610",
            size: expectedBytes,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const dst611 = host.device.createBuffer({
            label: "f3d-callback-dst-611",
            size: expectedBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          const dst612 = host.device.createBuffer({
            label: "f3d-callback-dst-612",
            size: expectedBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          cbBuffers = { 610: src, 611: dst611, 612: dst612 };

          prevWriteBuffer = hostObj.writeBuffer;
          prevCopyBuffer = hostObj.copyBufferToBuffer;

          hostObj.writeBuffer = (id, offset, bytes) => {
            const buf = cbBuffers?.[id];
            if (!buf) throw new Error(`f3dHost.writeBuffer: unknown buffer id ${id}`);
            host.device.queue.writeBuffer(buf, offset, bytes);
          };

          hostObj.copyBufferToBuffer = (srcId, srcOff, dstId, dstOff, size) => {
            const sBuf = cbBuffers?.[srcId];
            if (!sBuf) throw new Error(`f3dHost.copyBufferToBuffer: unknown source buffer ${srcId}`);
            const dBuf = cbBuffers?.[dstId];
            if (!dBuf) throw new Error(`f3dHost.copyBufferToBuffer: unknown destination buffer ${dstId}`);
            const encoder = host.device.createCommandEncoder();
            encoder.copyBufferToBuffer(sBuf, srcOff, dBuf, dstOff, size);
            host.device.queue.submit([encoder.finish()]);
          };
        },
        submitFrame: (floats, slot) => {
          const callbackCount = wasmExports.f3d_bridge_callback_storage_upload_frame(floats, recordCount, slot);
          if (callbackCount !== 2) {
            throw new Error(`f3d_bridge_callback_storage_upload_frame returned ${callbackCount}, expected 2`);
          }
          return host.device.queue.onSubmittedWorkDone();
        },
        readback: async (slot) => {
          await host.device.queue.onSubmittedWorkDone();
          const buf = cbBuffers?.[slot];
          if (!buf) throw new Error(`cbVariant readback: unknown slot ${slot}`);
          await buf.mapAsync(GPUMapMode.READ, 0, expectedBytes);
          const mapped = buf.getMappedRange(0, expectedBytes);
          const copy = new Uint8Array(mapped.slice(0));
          buf.unmap();
          return copy;
        },
        destroy: async () => {
          try {
            if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
          } finally {
            if (prevWriteBuffer !== undefined) hostObj.writeBuffer = prevWriteBuffer;
            else delete hostObj.writeBuffer;
            if (prevCopyBuffer !== undefined) hostObj.copyBufferToBuffer = prevCopyBuffer;
            else delete hostObj.copyBufferToBuffer;
            if (cbBuffers) {
              cbBuffers[610]?.destroy();
              cbBuffers[611]?.destroy();
              cbBuffers[612]?.destroy();
              cbBuffers = null;
            }
          }
        },
      };
    }

    case "workload_b_direct_js": {
      let directRunner = null;
      return {
        name: "workload_b_direct_js",
        implementation_owner: "Direct JS WebGPU",
        init: async () => {
          directRunner = createDirectPersistentPackedBufferRunner(host.device, recordCount);
        },
        submitFrame: (floats, slot) => {
          return directRunner.submitFrame(floats, null, slot);
        },
        readback: async (slot) => {
          return directRunner.readback(slot);
        },
        destroy: async () => {
          try {
            if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
          } finally {
            if (directRunner) {
              directRunner.destroy();
              directRunner = null;
            }
          }
        },
      };
    }

    case "workload_b_generated_js": {
      let genRunner = null;
      return {
        name: "workload_b_generated_js",
        implementation_owner: "Generated JS WebGPU submission",
        init: async () => {
          let genMod;
          try {
            genMod = await import("/out/browser-probe/static_packed_upload_4000.js");
          } catch (importErr) {
            throw new Error(`Failed to load /out/browser-probe/static_packed_upload_4000.js: ${importErr.message}`);
          }
          if (typeof genMod.initStaticPackedBuffers4000 !== "function") {
            throw new Error("Missing required export initStaticPackedBuffers4000 in static_packed_upload_4000.js");
          }
          if (typeof genMod.createStaticPackedUploadRunner4000 !== "function") {
            throw new Error("Missing required export createStaticPackedUploadRunner4000 in static_packed_upload_4000.js");
          }
          const buffers = genMod.initStaticPackedBuffers4000(host.device);
          genRunner = genMod.createStaticPackedUploadRunner4000(host.device, buffers);
        },
        submitFrame: (floats, slot) => {
          return genRunner.submitFrame(floats, null, slot);
        },
        readback: async (slot) => {
          return genRunner.readback(slot);
        },
        destroy: async () => {
          try {
            if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
          } finally {
            if (genRunner) {
              genRunner.destroy();
              genRunner = null;
            }
          }
        },
      };
    }

    default:
      throw new Error(`Unknown variant key: ${key}`);
  }
}

/**
 * Executes a credit loop for the default correctness mode (8 frames total, all recorded).
 */
async function runVariantCreditLoop(variant, upload0, upload1, expected0, expected1) {
  try {
    await variant.init();

    const inFlight = [];
    const perFrame = [];
    let maxInFlightObserved = 0;
    let firstError = null;
    let cpuDurationMs = 0;
    const tStart = performance.now();

    for (let i = 0; i < 8; i++) {
      if (firstError) break;

      if (inFlight.length >= 2) {
        await inFlight.shift();
        if (firstError) break;
      }

      if (inFlight.length + 1 > maxInFlightObserved) {
        maxInFlightObserved = inFlight.length + 1;
      }

      const slot = (i % 2 === 0) ? 611 : 612;
      const upload = (i % 2 === 0) ? upload0 : upload1;

      const tFrameStart = performance.now();
      let framePromise;
      try {
        framePromise = variant.submitFrame(upload.floats, slot);
      } catch (submitErr) {
        if (!firstError) firstError = submitErr;
        break;
      }
      const cpuMs = performance.now() - tFrameStart;
      cpuDurationMs += cpuMs;

      const tracked = framePromise.then(
        () => {
          const gpuElapsedMs = performance.now() - tFrameStart;
          perFrame.push({ frame: i, slot, cpu_ms: cpuMs, gpu_elapsed_ms: gpuElapsedMs });
        },
        (err) => {
          if (!firstError) firstError = err;
        }
      );
      inFlight.push(tracked);
    }

    await Promise.all(inFlight);
    const elapsedMs = performance.now() - tStart;
    if (firstError) throw firstError;

    // Correctness First: map-read 611 and 612 and compare with independently computed expected bytes
    const readback611 = await variant.readback(611);
    if (!readback611 || readback611.byteLength !== WORKLOAD_B_EXPECTED_BYTES) {
      throw new Error(`Slot 611 readback length mismatch: got ${readback611?.byteLength}, expected ${WORKLOAD_B_EXPECTED_BYTES}`);
    }
    assertBytesEqual(readback611, expected0, `${variant.name} readback 611`, 0);
    assertSpecialBits(readback611, `${variant.name} readback 611`, 0);

    const readback612 = await variant.readback(612);
    if (!readback612 || readback612.byteLength !== WORKLOAD_B_EXPECTED_BYTES) {
      throw new Error(`Slot 612 readback length mismatch: got ${readback612?.byteLength}, expected ${WORKLOAD_B_EXPECTED_BYTES}`);
    }
    assertBytesEqual(readback612, expected1, `${variant.name} readback 612`, 1);
    assertSpecialBits(readback612, `${variant.name} readback 612`, 1);

    return {
      status: "PASS",
      detail: `Workload (b) ${variant.name}: 8 frames (4000 records/192000B), max in-flight ${maxInFlightObserved}, cpu=${cpuDurationMs.toFixed(3)}ms elapsed=${elapsedMs.toFixed(3)}ms; readback 611 (upload 0) and 612 (upload 1) byte-exact with preserved -0.0 and quiet NaN`,
      implementation_owner: variant.implementation_owner,
      measurements: {
        implementation_owner: variant.implementation_owner,
        frame_count: 8,
        max_in_flight: maxInFlightObserved,
        cpu_duration_ms: cpuDurationMs,
        elapsed_ms: elapsedMs,
        per_frame: perFrame,
      },
    };
  } catch (err) {
    return {
      status: "FAIL",
      error: err.message || String(err),
      implementation_owner: variant.implementation_owner,
      measurements: null,
    };
  } finally {
    try {
      await variant.destroy();
    } catch (destroyErr) {
      console.warn(`Error during ${variant.name} cleanup:`, destroyErr);
    }
  }
}

/**
 * Runs all four variants of tmt.5 workload (b) in default correctness mode (§6.4, §16.7, §21):
 * 1. workload_b_bulk_packet
 * 2. workload_b_wasm_callbacks
 * 3. workload_b_direct_js
 * 4. workload_b_generated_js
 *
 * 8 frames per variant with max in-flight 2.
 */
export async function testWorkloadBVariants(host, wasmExports) {
  if (!host || !host.device) throw new Error("WebGpuBridgeHost device not initialized");

  const recordCount = WORKLOAD_B_RECORD_COUNT;
  const floatCount = recordCount * 12;

  const upload0 = generateDistinctAffineRows(recordCount, 0);
  const upload1 = generateDistinctAffineRows(recordCount, 1);
  const expected0 = buildExpectedBytes(upload0.buffer, floatCount);
  const expected1 = buildExpectedBytes(upload1.buffer, floatCount);

  const bulkVariant = createVariant("workload_b_bulk_packet", host, wasmExports);
  const cbVariant = createVariant("workload_b_wasm_callbacks", host, wasmExports);
  const directVariant = createVariant("workload_b_direct_js", host, wasmExports);
  const genVariant = createVariant("workload_b_generated_js", host, wasmExports);

  const variants = {
    workload_b_bulk_packet: await runVariantCreditLoop(bulkVariant, upload0, upload1, expected0, expected1),
    workload_b_wasm_callbacks: await runVariantCreditLoop(cbVariant, upload0, upload1, expected0, expected1),
    workload_b_direct_js: await runVariantCreditLoop(directVariant, upload0, upload1, expected0, expected1),
    workload_b_generated_js: await runVariantCreditLoop(genVariant, upload0, upload1, expected0, expected1),
  };

  const measurements = {
    workload_b_bulk_packet: variants.workload_b_bulk_packet.measurements,
    workload_b_wasm_callbacks: variants.workload_b_wasm_callbacks.measurements,
    workload_b_direct_js: variants.workload_b_direct_js.measurements,
    workload_b_generated_js: variants.workload_b_generated_js.measurements,
  };

  return { variants, measurements };
}

/**
 * Runs tmt.5 workload (b) in MEASUREMENT MODE (§6.4, §16.7, §21, bead f3d-03-reference-profiles-and-bridge-tmt.5):
 * - ROUNDS = 5
 * - WARMUP = 16 unrecorded frames per variant per round
 * - MEASURED = 240 recorded frames per variant per round
 * - Two-frame credit loop (max in-flight <= 2)
 * - Rotates variant order each round and records the order
 * - At end of each round: verifies readback 611 (upload 0) and 612 (upload 1) byte-exact with preserved -0.0 and quiet NaN
 * - Records raw per-frame rows for measured frames only, plus per-variant median and p95 of cpu_ms and gpu_elapsed_ms
 * - NO cross-variant ratios, rankings, or speedup wording
 */
export async function testWorkloadBMeasurementMode(host, wasmExports) {
  if (!host || !host.device) throw new Error("WebGpuBridgeHost device not initialized");

  const recordCount = WORKLOAD_B_RECORD_COUNT;
  const floatCount = recordCount * 12;
  const expectedBytes = WORKLOAD_B_EXPECTED_BYTES;

  const upload0 = generateDistinctAffineRows(recordCount, 0);
  const upload1 = generateDistinctAffineRows(recordCount, 1);
  const expected0 = buildExpectedBytes(upload0.buffer, floatCount);
  const expected1 = buildExpectedBytes(upload1.buffer, floatCount);

  const variantKeys = [
    "workload_b_bulk_packet",
    "workload_b_wasm_callbacks",
    "workload_b_direct_js",
    "workload_b_generated_js",
  ];

  // 1. COUNTS PASS: separate from and before timed rounds.
  // 8 untimed frames per variant with counting wrappers, strictly removed before timed rounds.
  async function runCountsPassForVariant(vKey) {
    const frameStatsList = [];
    const currentStats = { wasm_boundary_calls: 0, webgpu_api_calls: 0, js_bytes_copied: 0 };

    const origCreateCommandEncoder = host.device.createCommandEncoder;
    const countedQueue = host.device.queue;
    const origWriteBuffer = countedQueue?.writeBuffer;
    const origSubmit = countedQueue?.submit;
    const origOnSubmittedWorkDone = countedQueue?.onSubmittedWorkDone;

    const hadOwnCreateEncoder = Object.prototype.hasOwnProperty.call(host.device, "createCommandEncoder");
    const hadOwnWriteBuffer = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "writeBuffer") : false;
    const hadOwnSubmit = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "submit") : false;
    const hadOwnOnSubmittedWorkDone = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "onSubmittedWorkDone") : false;

    const globalScope = typeof window !== "undefined" ? window : globalThis;
    const hostObj = globalScope.f3dHost || (globalScope.f3dHost = {});

    // Create shadow wasmExports so module namespace object is never mutated
    const countingWasmExports = { ...wasmExports };

    host.device.createCommandEncoder = function(...args) {
      currentStats.webgpu_api_calls++;
      const encoder = origCreateCommandEncoder.apply(this, args);
      const origCopy = encoder.copyBufferToBuffer;
      const origFinish = encoder.finish;
      encoder.copyBufferToBuffer = function(...cArgs) {
        currentStats.webgpu_api_calls++;
        return origCopy.apply(this, cArgs);
      };
      encoder.finish = function(...fArgs) {
        currentStats.webgpu_api_calls++;
        return origFinish.apply(this, fArgs);
      };
      return encoder;
    };

    if (countedQueue) {
      countedQueue.writeBuffer = function(buf, offset, data, ...rest) {
        currentStats.webgpu_api_calls++;
        const byteLen = data?.byteLength || (data?.length ? data.length * (data.BYTES_PER_ELEMENT || 1) : 0);
        currentStats.js_bytes_copied += byteLen;
        return origWriteBuffer.call(this, buf, offset, data, ...rest);
      };

      countedQueue.submit = function(...args) {
        currentStats.webgpu_api_calls++;
        return origSubmit.apply(this, args);
      };

      countedQueue.onSubmittedWorkDone = function(...args) {
        currentStats.webgpu_api_calls++;
        return origOnSubmittedWorkDone.apply(this, args);
      };
    }

    if (typeof wasmExports?.f3d_build_affine_rows_storage_upload_frame_packet === "function") {
      countingWasmExports.f3d_build_affine_rows_storage_upload_frame_packet = function(...args) {
        currentStats.wasm_boundary_calls++;
        currentStats.js_bytes_copied += args[0].byteLength; // wasm-bindgen copies affine_rows into Wasm.
        const packet = wasmExports.f3d_build_affine_rows_storage_upload_frame_packet.apply(this, args);
        if (packet && packet.byteLength) {
          currentStats.js_bytes_copied += packet.byteLength;
        }
        return packet;
      };
    }

    if (typeof wasmExports?.f3d_bridge_callback_storage_upload_frame === "function") {
      countingWasmExports.f3d_bridge_callback_storage_upload_frame = function(...args) {
        currentStats.wasm_boundary_calls++;
        currentStats.js_bytes_copied += args[0].byteLength;
        return wasmExports.f3d_bridge_callback_storage_upload_frame.apply(this, args);
      };
    }

    const variant = createVariant(vKey, host, countingWasmExports, recordCount, expectedBytes);
    let origHostWrite = undefined;
    let origHostCopy = undefined;

    try {
      await variant.init();

      origHostWrite = hostObj.writeBuffer;
      origHostCopy = hostObj.copyBufferToBuffer;
      if (vKey === "workload_b_wasm_callbacks") {
        if (typeof origHostWrite === "function") {
          hostObj.writeBuffer = function(id, offset, bytes) {
            currentStats.wasm_boundary_calls++;
            if (bytes && bytes.byteLength) {
              currentStats.js_bytes_copied += bytes.byteLength;
            }
            return origHostWrite(id, offset, bytes);
          };
        }
        if (typeof origHostCopy === "function") {
          hostObj.copyBufferToBuffer = function(...cArgs) {
            currentStats.wasm_boundary_calls++;
            return origHostCopy.apply(this, cArgs);
          };
        }
      }

      for (let f = 0; f < 8; f++) {
        currentStats.wasm_boundary_calls = 0;
        currentStats.webgpu_api_calls = 0;
        currentStats.js_bytes_copied = 0;

        const slot = (f % 2 === 0 ? 611 : 612);
        const upload = (f % 2 === 0 ? upload0 : upload1);

        await variant.submitFrame(upload.floats, slot);

        frameStatsList.push({
          frame: f,
          queue_identity_stable: (host.device.queue === countedQueue),
          wasm_boundary_calls: currentStats.wasm_boundary_calls,
          webgpu_api_calls: currentStats.webgpu_api_calls,
          js_bytes_copied: currentStats.js_bytes_copied,
        });
      }
    } finally {
      if (hadOwnCreateEncoder) {
        host.device.createCommandEncoder = origCreateCommandEncoder;
      } else {
        delete host.device.createCommandEncoder;
      }

      if (countedQueue) {
        if (hadOwnWriteBuffer) {
          countedQueue.writeBuffer = origWriteBuffer;
        } else {
          delete countedQueue.writeBuffer;
        }

        if (hadOwnSubmit) {
          countedQueue.submit = origSubmit;
        } else {
          delete countedQueue.submit;
        }

        if (hadOwnOnSubmittedWorkDone) {
          countedQueue.onSubmittedWorkDone = origOnSubmittedWorkDone;
        } else {
          delete countedQueue.onSubmittedWorkDone;
        }
      }

      if (vKey === "workload_b_wasm_callbacks") {
        if (origHostWrite !== undefined) hostObj.writeBuffer = origHostWrite;
        else delete hostObj.writeBuffer;
        if (origHostCopy !== undefined) hostObj.copyBufferToBuffer = origHostCopy;
        else delete hostObj.copyBufferToBuffer;
      }

      try {
        await variant.destroy();
      } catch (_) {}
    }

    const first = frameStatsList[0] || { wasm_boundary_calls: 0, webgpu_api_calls: 0, js_bytes_copied: 0 };
    const framesDiffer = frameStatsList.some(
      f => f.wasm_boundary_calls !== first.wasm_boundary_calls ||
           f.webgpu_api_calls !== first.webgpu_api_calls ||
           f.js_bytes_copied !== first.js_bytes_copied
    );

    const unstableFrames = frameStatsList.filter(f => !f.queue_identity_stable);
    const hasUnstableQueue = unstableFrames.length > 0;
    let queueUnavailableReason = null;
    if (hasUnstableQueue) {
      const frameNames = unstableFrames.map(f => f.frame).join(", ");
      queueUnavailableReason = unstableFrames.length === 1
        ? `Queue identity unstable on frame ${frameNames} (host.device.queue !== countedQueue)`
        : `Queue identity unstable on frames ${frameNames} (host.device.queue !== countedQueue)`;
    }

    return {
      frames: frameStatsList,
      wasm_boundary_calls_per_frame: hasUnstableQueue ? null : first.wasm_boundary_calls,
      webgpu_api_calls_per_frame: hasUnstableQueue ? null : first.webgpu_api_calls,
      js_bytes_copied_per_frame: hasUnstableQueue ? null : first.js_bytes_copied,
      ...(queueUnavailableReason ? { unavailable_reason: queueUnavailableReason } : {}),
      frames_differ: framesDiffer,
      note: framesDiffer ? "frame counts differ across 8 untimed frames" : "all 8 frames identical",
    };
  }

  const countsMap = {};
  for (const vKey of variantKeys) {
    try {
      countsMap[vKey] = await runCountsPassForVariant(vKey);
    } catch (countsErr) {
      countsMap[vKey] = {
        error: countsErr.message || String(countsErr),
        wasm_boundary_calls_per_frame: null,
        webgpu_api_calls_per_frame: null,
        js_bytes_copied_per_frame: null,
        unavailable_reason: `Counts pass failed for ${vKey}: ${countsErr.message || String(countsErr)}`,
      };
    }
  }

  // 2. INIT TIME: 5 separate init passes with rotated variant order.
  // Each pass measures that variant's persistent init plus its first completed frame, then destroy.
  const totalInitPasses = 5;
  const initPasses = [];
  const initSamplesMap = {
    workload_b_bulk_packet: [],
    workload_b_wasm_callbacks: [],
    workload_b_direct_js: [],
    workload_b_generated_js: [],
  };
  const initErrors = {};

  for (let p = 0; p < totalInitPasses; p++) {
    const passOrder = [];
    for (let j = 0; j < variantKeys.length; j++) {
      passOrder.push(variantKeys[(p + j) % variantKeys.length]);
    }

    const passData = {
      pass_index: p,
      variant_order: [...passOrder],
      variants: {},
    };

    for (const vKey of passOrder) {
      const vInit = createVariant(vKey, host, wasmExports, recordCount, expectedBytes);
      try {
        const tInit0 = performance.now();
        await vInit.init();
        await vInit.submitFrame(upload0.floats, 611);
        const initDurationMs = performance.now() - tInit0;
        initSamplesMap[vKey].push(initDurationMs);
        passData.variants[vKey] = { init_ms: initDurationMs };
      } catch (initErr) {
        if (!initErrors[vKey]) {
          initErrors[vKey] = initErr.message || String(initErr);
        }
        passData.variants[vKey] = { error: initErr.message || String(initErr) };
      } finally {
        try {
          await vInit.destroy();
        } catch (_) {}
      }
    }

    initPasses.push(passData);
  }

  // Resource-timing entry for f3d_runtime_bg.wasm measures download duration (wasm_fetch_ms).
  // Wasm instantiate time is unavailable in this lane without modifying gpu_bridge_test.html.
  let wasmFetchMs = "unavailable";
  if (typeof performance !== "undefined" && typeof performance.getEntriesByType === "function") {
    const res = performance.getEntriesByType("resource")?.find(e => e.name?.includes("f3d_runtime_bg.wasm"));
    if (res?.duration != null) wasmFetchMs = res.duration;
  }
  const wasmInstantiateMs = "unavailable";

  // 3. TIMED ROUNDS: byte-for-byte same execution logic
  const roundRecords = [];
  const variantErrors = {};
  const allMeasuredRows = {
    workload_b_bulk_packet: [],
    workload_b_wasm_callbacks: [],
    workload_b_direct_js: [],
    workload_b_generated_js: [],
  };
  const allThroughputFps = {
    workload_b_bulk_packet: [],
    workload_b_wasm_callbacks: [],
    workload_b_direct_js: [],
    workload_b_generated_js: [],
  };

  const totalRounds = WORKLOAD_B_MEASURE_ROUNDS;
  const warmupCount = WORKLOAD_B_MEASURE_WARMUP;
  const measuredCount = WORKLOAD_B_MEASURE_MEASURED;
  const totalFrames = warmupCount + measuredCount; // 256 frames per variant per round

  for (let r = 0; r < totalRounds; r++) {
    // Rotate variant execution order each round
    const roundOrder = [];
    for (let j = 0; j < variantKeys.length; j++) {
      roundOrder.push(variantKeys[(r + j) % variantKeys.length]);
    }

    const roundData = {
      round_index: r,
      variant_order: [...roundOrder],
      variants: {},
    };

    for (let orderIdx = 0; orderIdx < roundOrder.length; orderIdx++) {
      const vKey = roundOrder[orderIdx];
      const variant = createVariant(vKey, host, wasmExports, recordCount, expectedBytes);

      try {
        await variant.init();

        const inFlight = [];
        const measuredFrameRows = [];
        let firstError = null;
        let cpuDurationMs = 0;
        let tMeasuredStart = 0;

        for (let f = 0; f < totalFrames; f++) {
          if (firstError) break;

          if (inFlight.length >= 2) {
            await inFlight.shift();
            if (firstError) break;
          }

          if (f === warmupCount) {
            tMeasuredStart = performance.now();
          }

          const slot = (f % 2 === 0 ? 611 : 612);
          const upload = (f % 2 === 0 ? upload0 : upload1);

          const isWarmup = f < warmupCount;
          const measuredIndex = f - warmupCount;

          const tFrameStart = performance.now();
          let framePromise;
          try {
            framePromise = variant.submitFrame(upload.floats, slot);
          } catch (submitErr) {
            if (!firstError) firstError = submitErr;
            break;
          }
          const cpuMs = performance.now() - tFrameStart;
          if (!isWarmup) {
            cpuDurationMs += cpuMs;
          }

          const tracked = framePromise.then(
            () => {
              if (!isWarmup) {
                const gpuElapsedMs = performance.now() - tFrameStart;
                measuredFrameRows.push({
                  frame: measuredIndex,
                  slot,
                  cpu_ms: cpuMs,
                  gpu_elapsed_ms: gpuElapsedMs,
                });
              }
            },
            (err) => {
              if (!firstError) firstError = err;
            }
          );
          inFlight.push(tracked);
        }

        await Promise.all(inFlight);
        const elapsedMs = tMeasuredStart > 0 ? (performance.now() - tMeasuredStart) : 0;
        if (firstError) throw firstError;

        const throughputFps = (elapsedMs > 0) ? ((measuredCount * 1000) / elapsedMs) : 0;
        allThroughputFps[vKey].push(throughputFps);

        // Drain GPU queue before readback
        if (host.device?.queue) {
          await host.device.queue.onSubmittedWorkDone();
        }

        // At end of round: map-read 611 and 612 per variant and compare exact bytes
        const rb611 = await variant.readback(611);
        if (!rb611 || rb611.byteLength !== expectedBytes) {
          throw new Error(`Round ${r} ${vKey} slot 611 readback length mismatch: got ${rb611?.byteLength}, expected ${expectedBytes}`);
        }
        assertBytesEqual(rb611, expected0, `${vKey} round ${r} readback 611`, 0);
        assertSpecialBits(rb611, `${vKey} round ${r} readback 611`, 0);

        const rb612 = await variant.readback(612);
        if (!rb612 || rb612.byteLength !== expectedBytes) {
          throw new Error(`Round ${r} ${vKey} slot 612 readback length mismatch: got ${rb612?.byteLength}, expected ${expectedBytes}`);
        }
        assertBytesEqual(rb612, expected1, `${vKey} round ${r} readback 612`, 1);
        assertSpecialBits(rb612, `${vKey} round ${r} readback 612`, 1);

        roundData.variants[vKey] = {
          order_index: orderIdx,
          warmup_count: warmupCount,
          measured_count: measuredCount,
          cpu_duration_ms: cpuDurationMs,
          elapsed_ms: elapsedMs,
          throughput_fps: throughputFps,
          per_frame: measuredFrameRows,
        };

        allMeasuredRows[vKey].push(...measuredFrameRows);
      } catch (err) {
        if (!variantErrors[vKey]) {
          variantErrors[vKey] = err;
        }
        roundData.variants[vKey] = {
          order_index: orderIdx,
          warmup_count: warmupCount,
          measured_count: 0,
          error: err.message || String(err),
          throughput_fps: 0,
          per_frame: [],
        };
      } finally {
        try {
          await variant.destroy();
        } catch (destroyErr) {
          console.warn(`Error destroying ${vKey} in round ${r}:`, destroyErr);
        }
      }
    }

    roundRecords.push(roundData);
  }

  // Compute per-variant summary
  const perVariantSummary = {};
  for (const vKey of variantKeys) {
    const rows = allMeasuredRows[vKey];
    const cpuVals = rows.map(r => r.cpu_ms);
    const gpuVals = rows.map(r => r.gpu_elapsed_ms);
    const fpsVals = allThroughputFps[vKey];
    const owner = {
      workload_b_bulk_packet: "Rust/Wasm packer and packet encoder; JavaScript WebGPU adapter",
      workload_b_wasm_callbacks: "Rust/Wasm callback loop; JavaScript WebGPU adapter",
      workload_b_direct_js: "Direct JS WebGPU",
      workload_b_generated_js: "Generated JS WebGPU submission",
    }[vKey];

    const countsEntry = countsMap[vKey];
    const countsValid = countsEntry && countsEntry.wasm_boundary_calls_per_frame != null;
    const wasmBoundaryCalls = countsValid ? countsEntry.wasm_boundary_calls_per_frame : null;
    const webgpuApiCalls = countsValid ? countsEntry.webgpu_api_calls_per_frame : null;
    const jsBytesCopied = countsValid ? countsEntry.js_bytes_copied_per_frame : null;

    const initSamples = initSamplesMap[vKey];
    const initValid = Array.isArray(initSamples) && initSamples.length > 0;
    const initMedian = initValid ? calculateMedian(initSamples) : null;
    const initP95 = initValid ? calculateP95(initSamples) : null;

    let unavailableReason = null;
    if (!countsValid && !initValid) {
      const countsReason = countsEntry?.unavailable_reason || `Counts pass failed (${countsEntry?.error || "unknown"})`;
      unavailableReason = `${countsReason}; Init passes failed (${initErrors[vKey] || "unknown"})`;
    } else if (!countsValid) {
      unavailableReason = countsEntry?.unavailable_reason || `Counts pass failed (${countsEntry?.error || "unknown"})`;
    } else if (!initValid) {
      unavailableReason = `Init passes failed (${initErrors[vKey] || "unknown"})`;
    }

    perVariantSummary[vKey] = {
      implementation_owner: owner,
      total_measured_frames: rows.length,
      init_ms: initMedian,
      init_ms_samples: initValid ? initSamples : null,
      init_ms_median: initMedian,
      init_ms_p95: initP95,
      init_order: initPasses.map(p => p.variant_order),
      wasm_boundary_calls_per_frame: wasmBoundaryCalls,
      webgpu_api_calls_per_frame: webgpuApiCalls,
      js_bytes_copied_per_frame: jsBytesCopied,
      ...(unavailableReason ? { unavailable_reason: unavailableReason } : {}),
      throughput_fps_median: calculateMedian(fpsVals),
      throughput_fps_p95: calculateP95(fpsVals),
      cpu_ms_median: calculateMedian(cpuVals),
      cpu_ms_p95: calculateP95(cpuVals),
      gpu_elapsed_ms_median: calculateMedian(gpuVals),
      gpu_elapsed_ms_p95: calculateP95(gpuVals),
    };
  }

  const measurements = {
    constants: {
      rounds: WORKLOAD_B_MEASURE_ROUNDS,
      warmup_per_variant: WORKLOAD_B_MEASURE_WARMUP,
      measured_per_variant: WORKLOAD_B_MEASURE_MEASURED,
      record_count: WORKLOAD_B_RECORD_COUNT,
      expected_bytes: WORKLOAD_B_EXPECTED_BYTES,
      init_passes: totalInitPasses,
      init_orders: initPasses.map(p => p.variant_order),
      wasm_fetch_ms: wasmFetchMs,
      wasm_instantiate_ms: wasmInstantiateMs,
      timing_notes: {
        cpu_ms: "runs from floats in to queue submitted",
        gpu_elapsed_ms: "runs from submit start to completion callback, including credit-loop wait",
        elapsed_ms: "round wall time over measured frames only",
        counts: "wasm_boundary_calls (JS-Wasm exports plus Wasm-JS callbacks), webgpu_api_calls (device, queue, command encoder, buffer methods), js_bytes_copied (JS-to-Wasm input copies, Wasm-to-JS output copies, and writeBuffer payloads; excludes Rust-internal copies) measured across 8 untimed frames per variant with counting wrappers removed before timed rounds",
        queue_identity_stable: "asserts host.device.queue === countedQueue on every counted frame; if false, per-frame counts are reported as null with unavailable_reason naming the frame",
        init_ms: "variant persistent init plus first completed frame across 5 separate passes with rotated variant order; records init_ms_samples (5 values), init_ms_median, init_ms_p95, and the init_order of every pass",
        wasm_fetch_ms: "resource-timing download duration for f3d_runtime_bg.wasm, or 'unavailable' if no resource timing entry matches",
        wasm_instantiate_ms: "unavailable in this lane without html changes",
        throughput_fps: "measured frames divided by measured-only elapsed seconds (measured_count * 1000 / elapsed_ms) per round",
      },
    },
    wasm_fetch_ms: wasmFetchMs,
    wasm_instantiate_ms: wasmInstantiateMs,
    init_passes: initPasses,
    counts: countsMap,
    rounds: roundRecords,
    per_variant_summary: perVariantSummary,
  };

  const variants = {};
  for (const vKey of variantKeys) {
    const owner = perVariantSummary[vKey].implementation_owner;
    const err = variantErrors[vKey];
    const summary = perVariantSummary[vKey];
    if (err) {
      variants[vKey] = {
        status: "FAIL",
        error: err.message || String(err),
        implementation_owner: owner,
      };
    } else {
      const initStr = summary.init_ms_median != null ? `${summary.init_ms_median.toFixed(3)}ms` : "unavailable";
      const countsStr = summary.wasm_boundary_calls_per_frame != null
        ? `wasm_calls=${summary.wasm_boundary_calls_per_frame}, webgpu_calls=${summary.webgpu_api_calls_per_frame}, bytes_copied=${summary.js_bytes_copied_per_frame}`
        : `counts unavailable (${summary.unavailable_reason || "unknown"})`;

      variants[vKey] = {
        status: "PASS",
        detail: `Workload (b) ${vKey}: ${totalRounds} rounds (${warmupCount} warmup + ${measuredCount} measured/round = ${summary.total_measured_frames} frames); readback 611 and 612 byte-exact with preserved -0.0 and quiet NaN; cpu median=${summary.cpu_ms_median.toFixed(3)}ms p95=${summary.cpu_ms_p95.toFixed(3)}ms; gpu median=${summary.gpu_elapsed_ms_median.toFixed(3)}ms p95=${summary.gpu_elapsed_ms_p95.toFixed(3)}ms; throughput fps median=${summary.throughput_fps_median.toFixed(2)}; init median=${initStr}; counts/frame: ${countsStr}; timing notes: cpu_ms runs from floats in to queue submitted; gpu_elapsed_ms runs from submit start to completion callback, including credit-loop wait; elapsed_ms is round wall time over measured frames only`,
        implementation_owner: owner,
      };
    }
  }

  return { variants, measurements };
}
