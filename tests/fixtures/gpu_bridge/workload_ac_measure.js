/**
 * workload_ac_measure.js - Production Measurement Harness for tmt.5 Workloads (a) and (c)
 * (§6.4, §16.7, §21, bead f3d-03-reference-profiles-and-bridge-tmt.5)
 *
 * Workload (a): 4000 dynamic-offset draw loop
 * - workload_a_direct: Direct JS WebGPU
 * - workload_a_bulk: Rust/Wasm packet encoder; JS WebGPU decoder
 * - workload_a_chatty: Rust/Wasm callback loop; JS WebGPU submission
 * - workload_a_generated: Rust/Wasm owned data packing; generated JS submission
 *
 * Workload (c): Render bundle execution (3996 bundle draws + 4 dynamic draws)
 * - workload_c_direct_bundle: Direct JS WebGPU (render bundle)
 * - workload_c_bulk_bundle: Rust/Wasm packet encoder; JS WebGPU decoder (render bundle)
 * - workload_c_chatty_bundle: Rust/Wasm callback loop; JS WebGPU submission (render bundle)
 * - workload_c_generated_bundle: Rust/Wasm owned data packing; generated JS submission (render bundle)
 *
 * Measurement Protocol:
 * 1. Untimed 8-frame counting pass per runner with real wrappers (wasm_boundary_calls, webgpu_api_calls, js_bytes_copied).
 *    Wrappers removed symmetrically in finally before timed frames. Chatty drawCall callbacks counted as Wasm boundary calls.
 * 2. 5 rotated init passes per runner: init plus first completed frame, destroy in finally, reporting samples, median, p95 and order.
 * 3. 5 rounds of 16 warmup + 240 measured frames under two-frame credit loop (max in-flight <= 2).
 *    Variant order rotated and recorded each round. Per-frame cpu_ms and gpu_elapsed_ms recorded for measured frames only.
 * 4. End-of-round pixel identity against direct reference (0 diffs and non-zero RGB assertion).
 * 5. Total lane wall time recorded.
 * 6. Detailed timing_notes for every field. Missing values are null with unavailable_reason, never numeric 0.
 * Zero speedup claims, ratios, or rankings.
 */

import { createDirectReferenceRenderer } from "./direct_reference.js";
import { currentPacketView } from "./borrowed_view.js";

export const BATCH_WIDTH = 256;
export const BATCH_HEIGHT = 256;
export const BATCH_DRAW_COUNT = 4000;
export const BATCH_PREFIX_COUNT = 3996; // 4000 - 4
export const BATCH_BYTES_PER_ROW = 1024;
export const BATCH_READBACK_SIZE = BATCH_BYTES_PER_ROW * BATCH_HEIGHT; // 262144 bytes

export const WORKLOAD_AC_MEASURE_ROUNDS = 5;
export const WORKLOAD_AC_MEASURE_WARMUP = 16;
export const WORKLOAD_AC_MEASURE_MEASURED = 240;
export const WORKLOAD_AC_MEASURE_TOTAL_FRAMES = 256; // 16 + 240
export const WORKLOAD_AC_MEASURE_INIT_PASSES = 5;

let cachedGenMod = null;
async function getGenMod() {
  if (!cachedGenMod) {
    cachedGenMod = await import("/out/browser-probe/static_submission_4000.js");
  }
  return cachedGenMod;
}

let cachedTailModule = null;
async function getTailModule() {
  if (!cachedTailModule) {
    cachedTailModule = await import("/out/browser-probe/static_submission_tail4.js");
  }
  return cachedTailModule;
}

export function buildSharedRows() {
  const sharedRows = new Float32Array(BATCH_DRAW_COUNT * 12);
  const cols = 80;
  const rows = 50;
  const sx = 0.018;
  const sy = 0.028;
  for (let i = 0; i < BATCH_DRAW_COUNT; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const tx = -0.95 + (col + 0.5) * (1.9 / cols);
    const ty = -0.95 + (row + 0.5) * (1.9 / rows);
    const off = i * 12;
    sharedRows[off]     = sx;  sharedRows[off + 1] = 0.0; sharedRows[off + 2]  = 0.0; sharedRows[off + 3]  = tx;
    sharedRows[off + 4] = 0.0; sharedRows[off + 5] = sy;  sharedRows[off + 6]  = 0.0; sharedRows[off + 7]  = ty;
    sharedRows[off + 8] = 0.0; sharedRows[off + 9] = 0.0; sharedRows[off + 10] = 1.0; sharedRows[off + 11] = 0.0;
  }
  return sharedRows;
}

export function buildRepeatedFrames(sharedRows, count = WORKLOAD_AC_MEASURE_TOTAL_FRAMES) {
  const repeatedFrames = [];
  for (let f = 0; f < count; f++) {
    const fRows = new Float32Array(sharedRows);
    const deltaX = (f % 2 === 0 ? 0.005 : -0.005) * ((f % 8) + 1);
    for (let i = 0; i < BATCH_DRAW_COUNT; i++) {
      fRows[i * 12 + 3] += deltaX;
    }
    repeatedFrames.push(fRows);
  }
  return repeatedFrames;
}

export function comparePixels(a, b) {
  if (!a || !b || a.byteLength !== b.byteLength) {
    throw new Error(`Pixel buffer length mismatch: ${a?.byteLength} vs ${b?.byteLength}`);
  }
  let diffs = 0;
  for (let i = 0; i < a.byteLength; i++) {
    if (a[i] !== b[i]) diffs++;
  }
  return diffs;
}

function calculateMedian(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function calculateP95(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const index = Math.ceil(0.95 * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
}

const settleAll = (promises) =>
  Promise.allSettled(promises).then((results) => {
    const failure = results.find((r) => r.status === "rejected");
    if (failure) throw failure.reason;
  });

/**
 * Creates runner instance for Workload A.
 */
function createWorkloadARunner(vKey, bridge, wasmExports, sharedRows, memory = null) {
  switch (vKey) {
    case "workload_a_direct": {
      let r = null;
      return {
        name: "workload_a_direct",
        implementation_owner: "Direct JS WebGPU",
        init: async () => {
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
        },
        submit: (rows) => {
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(rows, null, null);
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    case "workload_a_bulk": {
      return {
        name: "workload_a_bulk",
        implementation_owner: "Rust/Wasm packet encoder; JS WebGPU decoder",
        init: async () => {
          const initFn = wasmExports.f3d_build_affine_rows_batch_packet;
          if (typeof initFn !== "function") throw new Error("Missing f3d_build_affine_rows_batch_packet");
          const initBulkPacket = initFn(sharedRows, BATCH_WIDTH, BATCH_HEIGHT);
          if (!(initBulkPacket instanceof Uint8Array) || initBulkPacket.byteLength === 0) {
            throw new Error("Failed to generate initial bulk packet");
          }
          await bridge.executePacket(initBulkPacket);
        },
        submit: (rows) => {
          const framePacketFn = wasmExports.f3d_build_affine_rows_batch_frame_packet;
          if (typeof framePacketFn !== "function") throw new Error("Missing f3d_build_affine_rows_batch_frame_packet");
          const packet = framePacketFn(rows, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
          const packetCompletion = bridge.executePacket(packet);
          const queuePromise = bridge.device.queue.onSubmittedWorkDone();
          return settleAll([packetCompletion, queuePromise]);
        },
        readback: () => bridge.readbackBuffer(20, BATCH_READBACK_SIZE),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            for (const id of [1, 2, 20]) {
              bridge.buffers.get(id)?.destroy();
              bridge.buffers.delete(id);
              bridge.bufferEpochs?.delete(id);
            }
            bridge.textures.get(10)?.destroy();
            bridge.textures.delete(10);
            bridge.pipelines.delete(100);
          }
        },
      };
    }

    case "workload_a_bulk_borrowed": {
      let cachedView = null;
      let viewRebuilds = 0;
      return {
        name: "workload_a_bulk_borrowed",
        implementation_owner: "Rust/Wasm packet encoder (borrowed view); JS WebGPU decoder",
        getViewRebuilds: () => viewRebuilds,
        init: async () => {
          const initFn = wasmExports.f3d_build_affine_rows_batch_packet;
          if (typeof initFn !== "function") throw new Error("Missing f3d_build_affine_rows_batch_packet");
          const initBulkPacket = initFn(sharedRows, BATCH_WIDTH, BATCH_HEIGHT);
          if (!(initBulkPacket instanceof Uint8Array) || initBulkPacket.byteLength === 0) {
            throw new Error("Failed to generate initial bulk packet");
          }
          await bridge.executePacket(initBulkPacket);
        },
        submit: (rows) => {
          const framePacketBorrowedFn = wasmExports.f3d_build_affine_rows_batch_frame_packet_borrowed;
          if (typeof framePacketBorrowedFn !== "function") {
            throw new Error("Missing f3d_build_affine_rows_batch_frame_packet_borrowed");
          }
          const borrowEnterFn = wasmExports.f3d_borrow_enter;
          if (typeof borrowEnterFn !== "function") {
            throw new Error("Missing f3d_borrow_enter");
          }
          const borrowExitFn = wasmExports.f3d_borrow_exit;
          if (typeof borrowExitFn !== "function") {
            throw new Error("Missing f3d_borrow_exit");
          }

          const ptrLen = framePacketBorrowedFn(rows, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
          const ptr = ptrLen[0];
          const len = ptrLen[1];

          const token = borrowEnterFn();
          if (token === 0n || !token) {
            throw new Error("f3d_borrow_enter refused: borrow scope active or enter failed");
          }

          let completion;
          try {
            const view = currentPacketView(cachedView, memory, ptr, len);
            if (cachedView !== null && view !== cachedView) {
              viewRebuilds++;
            }
            cachedView = view;
            completion = bridge.executePacket(view);
          } finally {
            const exitOk = borrowExitFn(token);
            if (!exitOk) {
              throw new Error("f3d_borrow_exit failed: token invalid or borrow state mismatch");
            }
          }

          const queuePromise = bridge.device.queue.onSubmittedWorkDone();
          return settleAll([completion, queuePromise]);
        },
        readback: () => bridge.readbackBuffer(20, BATCH_READBACK_SIZE),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            for (const id of [1, 2, 20]) {
              bridge.buffers.get(id)?.destroy();
              bridge.buffers.delete(id);
              bridge.bufferEpochs?.delete(id);
            }
            bridge.textures.get(10)?.destroy();
            bridge.textures.delete(10);
            bridge.pipelines.delete(100);
          }
        },
      };
    }

    case "workload_a_chatty": {
      let r = null;
      const chattyDrawEncoder = (pass, bindGroup, count) => {
        const hostObj = typeof window !== "undefined" ? (window.f3dHost = window.f3dHost || {}) : (globalThis.f3dHost = globalThis.f3dHost || {});
        const prevDrawCall = hostObj.drawCall;
        try {
          hostObj.drawCall = (index) => {
            pass.setBindGroup(0, bindGroup, [index * 256]);
            pass.draw(3, 1, 0, 0);
          };
          const chattyFn = wasmExports.f3d_bridge_chatty_draw_loop;
          if (typeof chattyFn !== "function") throw new Error("Missing f3d_bridge_chatty_draw_loop");
          chattyFn(count);
        } finally {
          if (prevDrawCall !== undefined) hostObj.drawCall = prevDrawCall;
          else delete hostObj.drawCall;
        }
      };

      return {
        name: "workload_a_chatty",
        implementation_owner: "Rust/Wasm callback loop; JS WebGPU submission",
        init: async () => {
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
        },
        submit: (rows) => {
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(rows, chattyDrawEncoder, null);
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    case "workload_a_generated": {
      let r = null;
      let genMod = null;
      return {
        name: "workload_a_generated",
        implementation_owner: "Rust/Wasm owned data packing; generated JS submission",
        init: async () => {
          genMod = await getGenMod();
          if (typeof genMod.executeStaticSubmission4000 !== "function") {
            throw new Error("Missing executeStaticSubmission4000 in static_submission_4000.js");
          }
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
        },
        submit: (rows) => {
          const packFn = wasmExports.f3d_pack_affine_rows_bytes;
          if (typeof packFn !== "function") throw new Error("Missing f3d_pack_affine_rows_bytes");
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(
              null,
              (pass, bg) => genMod.executeStaticSubmission4000(pass, bg),
              () => packFn(rows)
            );
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    default:
      throw new Error(`Unknown Workload A runner key: ${vKey}`);
  }
}

/**
 * Creates runner instance for Workload C (render bundles).
 */
function createWorkloadCRunner(vKey, bridge, wasmExports, sharedRows) {
  const createBundleEncoder = (renderer, kind, tailMod) => {
    const recorder = bridge.device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
    recorder.setPipeline(renderer.pipeline);
    recorder.setVertexBuffer(0, renderer.vertexBuffer);
    for (let i = 0; i < BATCH_PREFIX_COUNT; i++) {
      recorder.setBindGroup(0, renderer.bindGroup, [i * 256]);
      recorder.draw(3, 1, 0, 0);
    }
    const bundle = recorder.finish();

    return (pass, bindGroup) => {
      pass.executeBundles([bundle]);
      pass.setPipeline(renderer.pipeline);
      pass.setVertexBuffer(0, renderer.vertexBuffer);
      if (kind === "generated") {
        tailMod.executeStaticSubmission4(pass, bindGroup);
      } else if (kind === "chatty") {
        const hostObj = typeof window !== "undefined" ? (window.f3dHost = window.f3dHost || {}) : (globalThis.f3dHost = globalThis.f3dHost || {});
        const previous = hostObj.drawCall;
        try {
          hostObj.drawCall = (index) => {
            pass.setBindGroup(0, bindGroup, [(BATCH_PREFIX_COUNT + index) * 256]);
            pass.draw(3, 1, 0, 0);
          };
          const chattyFn = wasmExports.f3d_bridge_chatty_draw_loop;
          if (typeof chattyFn !== "function") throw new Error("Missing f3d_bridge_chatty_draw_loop");
          chattyFn(4);
        } finally {
          if (previous === undefined) delete hostObj.drawCall;
          else hostObj.drawCall = previous;
        }
      } else {
        for (let i = BATCH_PREFIX_COUNT; i < BATCH_DRAW_COUNT; i++) {
          pass.setBindGroup(0, bindGroup, [i * 256]);
          pass.draw(3, 1, 0, 0);
        }
      }
    };
  };

  switch (vKey) {
    case "workload_c_direct_bundle": {
      let r = null;
      let bundleEncoder = null;
      return {
        name: "workload_c_direct_bundle",
        implementation_owner: "Direct JS WebGPU (render bundle)",
        init: async () => {
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
          bundleEncoder = createBundleEncoder(r, "direct", null);
        },
        submit: (rows) => {
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(rows, bundleEncoder, null);
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    case "workload_c_bulk_bundle": {
      let bulkBundle = null;
      return {
        name: "workload_c_bulk_bundle",
        implementation_owner: "Rust/Wasm packet encoder; JS WebGPU decoder (render bundle)",
        init: async () => {
          const initFn = wasmExports.f3d_build_affine_rows_batch_packet;
          if (typeof initFn !== "function") throw new Error("Missing f3d_build_affine_rows_batch_packet");
          const initBulkPacket = initFn(sharedRows, BATCH_WIDTH, BATCH_HEIGHT);
          if (!(initBulkPacket instanceof Uint8Array) || initBulkPacket.byteLength === 0) {
            throw new Error("Failed to generate initial bulk packet");
          }
          await bridge.executePacket(initBulkPacket);

          const bundlePacket = wasmExports.f3d_build_affine_rows_bundle_packet;
          if (typeof bundlePacket !== "function") throw new Error("Missing f3d_build_affine_rows_bundle_packet");
          const bundleInitPacket = bundlePacket(sharedRows, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT, 4, true);
          await bridge.executePacket(bundleInitPacket);
          bulkBundle = bridge.bundles.get(200);
          if (!bulkBundle) throw new Error("Rust bundle packet did not record bundle 200");
        },
        submit: (rows) => {
          if (bridge.bundles.get(200) !== bulkBundle) throw new Error("Bundle identity changed before replay");
          const bundlePacket = wasmExports.f3d_build_affine_rows_bundle_packet;
          const packet = bundlePacket(rows, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT, 4, false);
          return settleAll([bridge.executePacket(packet), bridge.device.queue.onSubmittedWorkDone()]);
        },
        readback: () => bridge.readbackBuffer(20, BATCH_READBACK_SIZE),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            bridge.bundles.delete(200);
            for (const id of [1, 2, 20]) {
              bridge.buffers.get(id)?.destroy();
              bridge.buffers.delete(id);
              bridge.bufferEpochs?.delete(id);
            }
            bridge.textures.get(10)?.destroy();
            bridge.textures.delete(10);
            bridge.pipelines.delete(100);
          }
        },
      };
    }

    case "workload_c_chatty_bundle": {
      let r = null;
      let bundleEncoder = null;
      return {
        name: "workload_c_chatty_bundle",
        implementation_owner: "Rust/Wasm callback loop; JS WebGPU submission (render bundle)",
        init: async () => {
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
          bundleEncoder = createBundleEncoder(r, "chatty", null);
        },
        submit: (rows) => {
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(rows, bundleEncoder, null);
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    case "workload_c_generated_bundle": {
      let r = null;
      let bundleEncoder = null;
      let tailMod = null;
      return {
        name: "workload_c_generated_bundle",
        implementation_owner: "Rust/Wasm owned data packing; generated JS submission (render bundle)",
        init: async () => {
          tailMod = await getTailModule();
          r = createDirectReferenceRenderer(bridge.device, BATCH_WIDTH, BATCH_HEIGHT, BATCH_DRAW_COUNT);
          bundleEncoder = createBundleEncoder(r, "generated", tailMod);
        },
        submit: (rows) => {
          const packFn = wasmExports.f3d_pack_affine_rows_bytes;
          if (typeof packFn !== "function") throw new Error("Missing f3d_pack_affine_rows_bytes");
          let donePromise = null;
          const scopePromise = bridge.withErrorScopes(["validation", "out-of-memory"], () => {
            donePromise = r.submitFrame(null, bundleEncoder, () => packFn(rows));
          });
          return settleAll([scopePromise, donePromise]);
        },
        readback: () => r.readback(),
        destroy: async () => {
          try {
            if (bridge.device?.queue) await bridge.device.queue.onSubmittedWorkDone();
          } finally {
            if (r) { r.destroy(); r = null; }
          }
        },
      };
    }

    default:
      throw new Error(`Unknown Workload C runner key: ${vKey}`);
  }
}

/**
 * Generic measurement engine for Workloads (a) and (c).
 */
async function runWorkloadMeasurement(config) {
  const {
    laneName,
    workloadLabel,
    variantKeys,
    directKey,
    createRunnerFn,
    bridge,
    wasmExports,
    sharedRows,
    repeatedFrames,
  } = config;

  const tLaneStart = performance.now();
  const isBorrowedLane = laneName === "borrowed_bulk_measure";

  // 1. COUNTS PASS: separate from and before timed rounds.
  // 8 untimed frames per runner with counting wrappers, strictly removed before timed rounds.
  async function runCountsPassForRunner(vKey) {
    const frameStatsList = [];
    const currentStats = {
      wasm_boundary_calls: 0,
      webgpu_api_calls: 0,
      js_bytes_copied: 0,
      ...(isBorrowedLane ? { js_bytes_viewed: 0 } : {}),
    };

    const origCreateCommandEncoder = bridge.device.createCommandEncoder;
    const origCreateRenderBundleEncoder = bridge.device.createRenderBundleEncoder;
    const countedQueue = bridge.device.queue;
    const origWriteBuffer = countedQueue?.writeBuffer;
    const origSubmit = countedQueue?.submit;
    const origOnSubmittedWorkDone = countedQueue?.onSubmittedWorkDone;

    const hadOwnCreateEncoder = Object.prototype.hasOwnProperty.call(bridge.device, "createCommandEncoder");
    const hadOwnCreateBundleEncoder = Object.prototype.hasOwnProperty.call(bridge.device, "createRenderBundleEncoder");
    const hadOwnWriteBuffer = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "writeBuffer") : false;
    const hadOwnSubmit = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "submit") : false;
    const hadOwnOnSubmittedWorkDone = countedQueue ? Object.prototype.hasOwnProperty.call(countedQueue, "onSubmittedWorkDone") : false;

    const countingWasmExports = { ...wasmExports };

    bridge.device.createCommandEncoder = function (...args) {
      currentStats.webgpu_api_calls++;
      const encoder = origCreateCommandEncoder.apply(this, args);
      const origCopy = encoder.copyBufferToBuffer;
      const origCopyTex = encoder.copyTextureToBuffer;
      const origFinish = encoder.finish;
      const origBeginPass = encoder.beginRenderPass;

      if (origCopy) {
        encoder.copyBufferToBuffer = function (...cArgs) {
          currentStats.webgpu_api_calls++;
          return origCopy.apply(this, cArgs);
        };
      }
      if (origCopyTex) {
        encoder.copyTextureToBuffer = function (...tArgs) {
          currentStats.webgpu_api_calls++;
          return origCopyTex.apply(this, tArgs);
        };
      }
      if (origFinish) {
        encoder.finish = function (...fArgs) {
          currentStats.webgpu_api_calls++;
          return origFinish.apply(this, fArgs);
        };
      }
      if (origBeginPass) {
        encoder.beginRenderPass = function (...pArgs) {
          currentStats.webgpu_api_calls++;
          const pass = origBeginPass.apply(this, pArgs);
          const origSetPipeline = pass.setPipeline;
          const origSetVb = pass.setVertexBuffer;
          const origSetBg = pass.setBindGroup;
          const origDraw = pass.draw;
          const origExecBundles = pass.executeBundles;
          const origEnd = pass.end;

          if (origSetPipeline) {
            pass.setPipeline = function (...a) {
              currentStats.webgpu_api_calls++;
              return origSetPipeline.apply(this, a);
            };
          }
          if (origSetVb) {
            pass.setVertexBuffer = function (...a) {
              currentStats.webgpu_api_calls++;
              return origSetVb.apply(this, a);
            };
          }
          if (origSetBg) {
            pass.setBindGroup = function (...a) {
              currentStats.webgpu_api_calls++;
              return origSetBg.apply(this, a);
            };
          }
          if (origDraw) {
            pass.draw = function (...a) {
              currentStats.webgpu_api_calls++;
              return origDraw.apply(this, a);
            };
          }
          if (origExecBundles) {
            pass.executeBundles = function (...a) {
              currentStats.webgpu_api_calls++;
              return origExecBundles.apply(this, a);
            };
          }
          if (origEnd) {
            pass.end = function (...a) {
              currentStats.webgpu_api_calls++;
              return origEnd.apply(this, a);
            };
          }
          return pass;
        };
      }
      return encoder;
    };

    if (origCreateRenderBundleEncoder) {
      bridge.device.createRenderBundleEncoder = function (...args) {
        currentStats.webgpu_api_calls++;
        const rbe = origCreateRenderBundleEncoder.apply(this, args);
        const origRbeSetPipeline = rbe.setPipeline;
        const origRbeSetVb = rbe.setVertexBuffer;
        const origRbeSetBg = rbe.setBindGroup;
        const origRbeDraw = rbe.draw;
        const origRbeFinish = rbe.finish;

        if (origRbeSetPipeline) {
          rbe.setPipeline = function (...a) {
            currentStats.webgpu_api_calls++;
            return origRbeSetPipeline.apply(this, a);
          };
        }
        if (origRbeSetVb) {
          rbe.setVertexBuffer = function (...a) {
            currentStats.webgpu_api_calls++;
            return origRbeSetVb.apply(this, a);
          };
        }
        if (origRbeSetBg) {
          rbe.setBindGroup = function (...a) {
            currentStats.webgpu_api_calls++;
            return origRbeSetBg.apply(this, a);
          };
        }
        if (origRbeDraw) {
          rbe.draw = function (...a) {
            currentStats.webgpu_api_calls++;
            return origRbeDraw.apply(this, a);
          };
        }
        if (origRbeFinish) {
          rbe.finish = function (...a) {
            currentStats.webgpu_api_calls++;
            return origRbeFinish.apply(this, a);
          };
        }
        return rbe;
      };
    }

    if (countedQueue) {
      countedQueue.writeBuffer = function (buf, offset, data, ...rest) {
        currentStats.webgpu_api_calls++;
        const byteLen = data?.byteLength || (data?.length ? data.length * (data.BYTES_PER_ELEMENT || 1) : 0);
        currentStats.js_bytes_copied += byteLen;
        return origWriteBuffer.call(this, buf, offset, data, ...rest);
      };

      countedQueue.submit = function (...args) {
        currentStats.webgpu_api_calls++;
        return origSubmit.apply(this, args);
      };

      countedQueue.onSubmittedWorkDone = function (...args) {
        currentStats.webgpu_api_calls++;
        return origOnSubmittedWorkDone.apply(this, args);
      };
    }

    if (typeof wasmExports?.f3d_build_affine_rows_batch_frame_packet === "function") {
      countingWasmExports.f3d_build_affine_rows_batch_frame_packet = function (...args) {
        currentStats.wasm_boundary_calls++;
        const packet = wasmExports.f3d_build_affine_rows_batch_frame_packet.apply(this, args);
        if (packet && packet.byteLength) currentStats.js_bytes_copied += packet.byteLength;
        return packet;
      };
    }

    if (typeof wasmExports?.f3d_build_affine_rows_bundle_packet === "function") {
      countingWasmExports.f3d_build_affine_rows_bundle_packet = function (...args) {
        currentStats.wasm_boundary_calls++;
        const packet = wasmExports.f3d_build_affine_rows_bundle_packet.apply(this, args);
        if (packet && packet.byteLength) currentStats.js_bytes_copied += packet.byteLength;
        return packet;
      };
    }

    if (typeof wasmExports?.f3d_pack_affine_rows_bytes === "function") {
      countingWasmExports.f3d_pack_affine_rows_bytes = function (...args) {
        currentStats.wasm_boundary_calls++;
        const bytes = wasmExports.f3d_pack_affine_rows_bytes.apply(this, args);
        if (bytes && bytes.byteLength) currentStats.js_bytes_copied += bytes.byteLength;
        return bytes;
      };
    }

    if (typeof wasmExports?.f3d_bridge_chatty_draw_loop === "function") {
      countingWasmExports.f3d_bridge_chatty_draw_loop = function (count) {
        currentStats.wasm_boundary_calls++; // JS-to-Wasm export call
        const hostObj = typeof window !== "undefined" ? (window.f3dHost = window.f3dHost || {}) : (globalThis.f3dHost = globalThis.f3dHost || {});
        const origDrawCall = hostObj.drawCall;
        if (typeof origDrawCall === "function") {
          hostObj.drawCall = function (idx) {
            currentStats.wasm_boundary_calls++; // Wasm-to-JS callback
            return origDrawCall(idx);
          };
        }
        try {
          return wasmExports.f3d_bridge_chatty_draw_loop.call(this, count);
        } finally {
          if (origDrawCall !== undefined) hostObj.drawCall = origDrawCall;
          else delete hostObj.drawCall;
        }
      };
    }

    if (typeof wasmExports?.f3d_build_affine_rows_batch_frame_packet_borrowed === "function") {
      countingWasmExports.f3d_build_affine_rows_batch_frame_packet_borrowed = function (...args) {
        currentStats.wasm_boundary_calls++;
        const ptrLen = wasmExports.f3d_build_affine_rows_batch_frame_packet_borrowed.apply(this, args);
        currentStats.js_bytes_copied += 8;
        if (ptrLen && ptrLen.length >= 2) {
          currentStats.js_bytes_viewed = (currentStats.js_bytes_viewed || 0) + ptrLen[1];
        }
        return ptrLen;
      };
    }

    if (typeof wasmExports?.f3d_borrow_enter === "function") {
      countingWasmExports.f3d_borrow_enter = function (...args) {
        currentStats.wasm_boundary_calls++;
        return wasmExports.f3d_borrow_enter.apply(this, args);
      };
    }

    if (typeof wasmExports?.f3d_borrow_exit === "function") {
      countingWasmExports.f3d_borrow_exit = function (...args) {
        currentStats.wasm_boundary_calls++;
        return wasmExports.f3d_borrow_exit.apply(this, args);
      };
    }

    const runner = createRunnerFn(vKey, bridge, countingWasmExports, sharedRows);
    try {
      await runner.init();

      for (let f = 0; f < 8; f++) {
        currentStats.wasm_boundary_calls = 0;
        currentStats.webgpu_api_calls = 0;
        currentStats.js_bytes_copied = 0;
        if (isBorrowedLane) {
          currentStats.js_bytes_viewed = 0;
        }

        const rebuildsBefore = typeof runner.getViewRebuilds === "function" ? runner.getViewRebuilds() : 0;
        await runner.submit(repeatedFrames[f]);
        const rebuildsAfter = typeof runner.getViewRebuilds === "function" ? runner.getViewRebuilds() : 0;
        const frameRebuilds = rebuildsAfter - rebuildsBefore;

        const frameStat = {
          frame: f,
          queue_identity_stable: bridge.device.queue === countedQueue,
          wasm_boundary_calls: currentStats.wasm_boundary_calls,
          webgpu_api_calls: currentStats.webgpu_api_calls,
          js_bytes_copied: currentStats.js_bytes_copied,
        };
        if (isBorrowedLane) {
          frameStat.js_bytes_viewed = currentStats.js_bytes_viewed;
          frameStat.view_rebuilds = frameRebuilds;
        }
        frameStatsList.push(frameStat);
      }
    } finally {
      if (hadOwnCreateEncoder) bridge.device.createCommandEncoder = origCreateCommandEncoder;
      else delete bridge.device.createCommandEncoder;

      if (hadOwnCreateBundleEncoder) bridge.device.createRenderBundleEncoder = origCreateRenderBundleEncoder;
      else delete bridge.device.createRenderBundleEncoder;

      if (countedQueue) {
        if (hadOwnWriteBuffer) countedQueue.writeBuffer = origWriteBuffer;
        else delete countedQueue.writeBuffer;

        if (hadOwnSubmit) countedQueue.submit = origSubmit;
        else delete countedQueue.submit;

        if (hadOwnOnSubmittedWorkDone) countedQueue.onSubmittedWorkDone = origOnSubmittedWorkDone;
        else delete countedQueue.onSubmittedWorkDone;
      }

      try {
        await runner.destroy();
      } catch (_) {}
    }

    const first = frameStatsList[0] || {
      wasm_boundary_calls: 0,
      webgpu_api_calls: 0,
      js_bytes_copied: 0,
      ...(isBorrowedLane ? { js_bytes_viewed: 0, view_rebuilds: 0 } : {}),
    };
    const framesDiffer = frameStatsList.some(
      (f) =>
        f.wasm_boundary_calls !== first.wasm_boundary_calls ||
        f.webgpu_api_calls !== first.webgpu_api_calls ||
        f.js_bytes_copied !== first.js_bytes_copied ||
        (isBorrowedLane &&
          (f.js_bytes_viewed !== first.js_bytes_viewed || f.view_rebuilds !== first.view_rebuilds))
    );

    let viewRebuildsPerFrame = null;
    let viewRebuildsUnavailableReason = null;
    if (isBorrowedLane) {
      const steadyRebuilds = frameStatsList.slice(1).map((f) => f.view_rebuilds ?? 0);
      if (steadyRebuilds.length === 7 && steadyRebuilds.every((v) => v === steadyRebuilds[0])) {
        viewRebuildsPerFrame = steadyRebuilds[0];
      } else {
        viewRebuildsUnavailableReason = `Steady-state view rebuilds differ across frames 1..7: [${steadyRebuilds.join(", ")}]`;
      }
    }

    const unstableFrames = frameStatsList.filter((f) => !f.queue_identity_stable);
    const hasUnstableQueue = unstableFrames.length > 0;
    let queueUnavailableReason = null;
    if (hasUnstableQueue) {
      const frameNames = unstableFrames.map((f) => f.frame).join(", ");
      queueUnavailableReason = unstableFrames.length === 1
        ? `Queue identity unstable on frame ${frameNames} (bridge.device.queue !== countedQueue)`
        : `Queue identity unstable on frames ${frameNames} (bridge.device.queue !== countedQueue)`;
    }

    const countsUnavailableReason = queueUnavailableReason
      ? (viewRebuildsUnavailableReason ? `${queueUnavailableReason}; ${viewRebuildsUnavailableReason}` : queueUnavailableReason)
      : viewRebuildsUnavailableReason;

    return {
      frames: frameStatsList,
      wasm_boundary_calls_per_frame: hasUnstableQueue ? null : first.wasm_boundary_calls,
      webgpu_api_calls_per_frame: hasUnstableQueue ? null : first.webgpu_api_calls,
      js_bytes_copied_per_frame: hasUnstableQueue ? null : first.js_bytes_copied,
      ...(isBorrowedLane
        ? {
            js_bytes_viewed_per_frame: hasUnstableQueue ? null : first.js_bytes_viewed,
            js_bytes_viewed_frames: frameStatsList.map((f) => f.js_bytes_viewed ?? 0),
            view_rebuilds_per_frame: hasUnstableQueue ? null : viewRebuildsPerFrame,
            view_rebuilds_frames: frameStatsList.map((f) => f.view_rebuilds ?? 0),
          }
        : {}),
      ...(countsUnavailableReason ? { unavailable_reason: countsUnavailableReason } : {}),
      frames_differ: framesDiffer,
      note: framesDiffer ? "frame counts differ across 8 untimed frames" : "all 8 frames identical",
    };
  }

  const countsMap = {};
  for (const vKey of variantKeys) {
    try {
      countsMap[vKey] = await runCountsPassForRunner(vKey);
    } catch (countsErr) {
      countsMap[vKey] = {
        error: countsErr.message || String(countsErr),
        wasm_boundary_calls_per_frame: null,
        webgpu_api_calls_per_frame: null,
        js_bytes_copied_per_frame: null,
        ...(isBorrowedLane
          ? {
              js_bytes_viewed_per_frame: null,
              js_bytes_viewed_frames: null,
              view_rebuilds_per_frame: null,
              view_rebuilds_frames: null,
            }
          : {}),
        unavailable_reason: `Counts pass failed for ${vKey}: ${countsErr.message || String(countsErr)}`,
      };
    }
  }

  // 2. INIT TIME: 5 separate init passes with rotated variant order.
  // Each pass measures that runner's persistent init plus its first completed frame, then destroy.
  const totalInitPasses = WORKLOAD_AC_MEASURE_INIT_PASSES;
  const initPasses = [];
  const initSamplesMap = {};
  for (const vKey of variantKeys) {
    initSamplesMap[vKey] = [];
  }
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
      const vInit = createRunnerFn(vKey, bridge, wasmExports, sharedRows);
      try {
        const tInit0 = performance.now();
        await vInit.init();
        await vInit.submit(repeatedFrames[0]);
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
  let wasmFetchMs = "unavailable";
  if (typeof performance !== "undefined" && typeof performance.getEntriesByType === "function") {
    const res = performance.getEntriesByType("resource")?.find((e) => e.name?.includes("f3d_runtime_bg.wasm"));
    if (res?.duration != null) wasmFetchMs = res.duration;
  }
  const wasmInstantiateMs = "unavailable";

  // 3. TIMED ROUNDS: 5 rounds of 16 warmup + 240 measured frames with rotated order
  const roundRecords = [];
  const variantErrors = {};
  const allMeasuredRows = {};
  const allThroughputFps = {};
  for (const vKey of variantKeys) {
    allMeasuredRows[vKey] = [];
    allThroughputFps[vKey] = [];
  }

  const totalRounds = WORKLOAD_AC_MEASURE_ROUNDS;
  const warmupCount = WORKLOAD_AC_MEASURE_WARMUP;
  const measuredCount = WORKLOAD_AC_MEASURE_MEASURED;
  const totalFrames = warmupCount + measuredCount; // 256

  for (let r = 0; r < totalRounds; r++) {
    const roundOrder = [];
    for (let j = 0; j < variantKeys.length; j++) {
      roundOrder.push(variantKeys[(r + j) % variantKeys.length]);
    }

    const roundData = {
      round_index: r,
      variant_order: [...roundOrder],
      variants: {},
    };

    const roundPixels = {};

    for (let orderIdx = 0; orderIdx < roundOrder.length; orderIdx++) {
      const vKey = roundOrder[orderIdx];
      const runner = createRunnerFn(vKey, bridge, wasmExports, sharedRows);

      try {
        await runner.init();

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

          const isWarmup = f < warmupCount;
          const measuredIndex = f - warmupCount;

          const tFrameStart = performance.now();
          let framePromise;
          try {
            framePromise = runner.submit(repeatedFrames[f]);
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
        const elapsedMs = tMeasuredStart > 0 ? performance.now() - tMeasuredStart : 0;
        if (firstError) throw firstError;

        const throughputFps = elapsedMs > 0 ? (measuredCount * 1000) / elapsedMs : 0;
        allThroughputFps[vKey].push(throughputFps);

        // Readback at end of round
        const finalPixels = await runner.readback();
        if (!finalPixels || finalPixels.byteLength !== BATCH_READBACK_SIZE) {
          throw new Error(
            `Round ${r} ${vKey} readback length mismatch: got ${finalPixels?.byteLength}, expected ${BATCH_READBACK_SIZE}`
          );
        }
        roundPixels[vKey] = finalPixels;

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
          await runner.destroy();
        } catch (destroyErr) {
          console.warn(`Error destroying ${vKey} in round ${r}:`, destroyErr);
        }
      }
    }

    // (4) End-of-round pixel identity against direct reference
    const directPixels = roundPixels[directKey];
    if (directPixels) {
      let directNonZeroRgb = 0;
      for (let i = 0; i < directPixels.byteLength; i++) {
        if (i % 4 !== 3 && directPixels[i] !== 0) directNonZeroRgb++;
      }
      if (directNonZeroRgb === 0) {
        const noRgbErr = new Error(`Direct reference rendered image in round ${r} has no non-zero RGB color channels`);
        if (!variantErrors[directKey]) variantErrors[directKey] = noRgbErr;
      }

      for (const vKey of variantKeys) {
        if (vKey === directKey) continue;
        const vPixels = roundPixels[vKey];
        if (vPixels) {
          const diffs = comparePixels(vPixels, directPixels);
          if (diffs > 0) {
            const mismatchErr = new Error(`Round ${r} ${vKey} pixels mismatch direct reference: ${diffs} differences`);
            if (!variantErrors[vKey]) variantErrors[vKey] = mismatchErr;
          }
        }
      }
    } else if (!variantErrors[directKey]) {
      variantErrors[directKey] = new Error(`Direct reference missing readback pixels in round ${r}`);
    }

    roundRecords.push(roundData);
  }

  // Compute per-variant summary
  const perVariantSummary = {};
  for (const vKey of variantKeys) {
    const rows = allMeasuredRows[vKey];
    const cpuVals = rows.map((r) => r.cpu_ms);
    const gpuVals = rows.map((r) => r.gpu_elapsed_ms);
    const fpsVals = allThroughputFps[vKey];

    const dummyRunner = createRunnerFn(vKey, bridge, wasmExports, sharedRows);
    const owner = dummyRunner.implementation_owner;

    const countsEntry = countsMap[vKey];
    const countsValid = countsEntry && countsEntry.wasm_boundary_calls_per_frame != null;
    const wasmBoundaryCalls = countsValid ? countsEntry.wasm_boundary_calls_per_frame : null;
    const webgpuApiCalls = countsValid ? countsEntry.webgpu_api_calls_per_frame : null;
    const jsBytesCopied = countsValid ? countsEntry.js_bytes_copied_per_frame : null;
    const jsBytesViewed =
      isBorrowedLane && countsValid && countsEntry.js_bytes_viewed_per_frame != null
        ? countsEntry.js_bytes_viewed_per_frame
        : null;
    const viewRebuilds =
      isBorrowedLane && countsValid && countsEntry.view_rebuilds_per_frame != null
        ? countsEntry.view_rebuilds_per_frame
        : null;

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
    } else if (isBorrowedLane && countsValid && countsEntry.view_rebuilds_per_frame == null && countsEntry.unavailable_reason) {
      unavailableReason = countsEntry.unavailable_reason;
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
      init_order: initPasses.map((p) => p.variant_order),
      wasm_boundary_calls_per_frame: wasmBoundaryCalls,
      webgpu_api_calls_per_frame: webgpuApiCalls,
      js_bytes_copied_per_frame: jsBytesCopied,
      ...(isBorrowedLane
        ? {
            js_bytes_viewed_per_frame: jsBytesViewed,
            view_rebuilds_per_frame: viewRebuilds,
          }
        : {}),
      ...(unavailableReason ? { unavailable_reason: unavailableReason } : {}),
      throughput_fps_median: calculateMedian(fpsVals),
      throughput_fps_p95: calculateP95(fpsVals),
      cpu_ms_median: calculateMedian(cpuVals),
      cpu_ms_p95: calculateP95(cpuVals),
      gpu_elapsed_ms_median: calculateMedian(gpuVals),
      gpu_elapsed_ms_p95: calculateP95(gpuVals),
    };
  }

  const totalWallTimeMs = performance.now() - tLaneStart;

  const measurements = {
    constants: {
      rounds: WORKLOAD_AC_MEASURE_ROUNDS,
      warmup_per_variant: WORKLOAD_AC_MEASURE_WARMUP,
      measured_per_variant: WORKLOAD_AC_MEASURE_MEASURED,
      draw_count: BATCH_DRAW_COUNT,
      init_passes: totalInitPasses,
      init_orders: initPasses.map((p) => p.variant_order),
      wasm_fetch_ms: wasmFetchMs,
      wasm_instantiate_ms: wasmInstantiateMs,
      total_wall_time_ms: totalWallTimeMs,
      timing_notes: {
        cpu_ms: "runs from frame rows in to queue submitted",
        gpu_elapsed_ms: "runs from submit start to completion callback, including credit-loop wait",
        elapsed_ms: "round wall time over measured frames only",
        counts: isBorrowedLane
          ? "wasm_boundary_calls (JS-Wasm exports plus Wasm-JS callbacks, including chatty drawCall callbacks and borrow enter/exit), webgpu_api_calls (device, queue, encoder, pass methods), js_bytes_copied (Wasm-JS boundary copies plus writeBuffer payloads), js_bytes_viewed (direct Wasm memory packet bytes read by JS decoder), and view_rebuilds (Uint8Array view reallocations across frames) measured across 8 untimed frames per runner with counting wrappers removed before timed rounds"
          : "wasm_boundary_calls (JS-Wasm exports plus Wasm-JS callbacks, including chatty drawCall callbacks), webgpu_api_calls (device, queue, encoder, pass methods), js_bytes_copied (Wasm-JS boundary copies plus writeBuffer payloads) measured across 8 untimed frames per runner with counting wrappers removed before timed rounds",
        queue_identity_stable: "asserts bridge.device.queue === countedQueue on every counted frame; if false, per-frame counts are reported as null with unavailable_reason naming the frame",
        ...(isBorrowedLane
          ? {
              bytes_viewed: "bytes_viewed is not zero-copy to the GPU (queue.writeBuffer still copies)",
              borrow_scope: "BorrowScope growth refusal is bookkeeping only (Rust try_grow_memory does not call memory.grow), so allocator growth is caught only by the JS buffer-identity check",
              view_rebuilds: "frame 0 has no cached view by construction, so rebuilds are counted from frame 1",
            }
          : {}),
        init_ms: "runner persistent init plus first completed frame across 5 separate passes with rotated runner order; records init_ms_samples (5 values), init_ms_median, init_ms_p95, and the init_order of every pass",
        wasm_fetch_ms: "resource-timing download duration for f3d_runtime_bg.wasm, or 'unavailable' if no resource timing entry matches",
        wasm_instantiate_ms: "unavailable in this lane without html changes",
        throughput_fps: "measured frames divided by measured-only elapsed seconds (measured_count * 1000 / elapsed_ms) per round",
        total_wall_time_ms: "total wall clock elapsed time for the measurement lane from lane start to completion",
      },
    },
    wasm_fetch_ms: wasmFetchMs,
    wasm_instantiate_ms: wasmInstantiateMs,
    total_wall_time_ms: totalWallTimeMs,
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
      let countsStr;
      if (summary.wasm_boundary_calls_per_frame != null) {
        countsStr = `wasm_calls=${summary.wasm_boundary_calls_per_frame}, webgpu_calls=${summary.webgpu_api_calls_per_frame}, bytes_copied=${summary.js_bytes_copied_per_frame}`;
        if (isBorrowedLane) {
          countsStr += `, bytes_viewed=${summary.js_bytes_viewed_per_frame}, view_rebuilds=${summary.view_rebuilds_per_frame}`;
        }
      } else {
        countsStr = `counts unavailable (${summary.unavailable_reason || "unknown"})`;
      }

      variants[vKey] = {
        status: "PASS",
        detail: `Workload ${workloadLabel} ${vKey}: ${totalRounds} rounds (${warmupCount} warmup + ${measuredCount} measured/round = ${summary.total_measured_frames} frames); pixel-identical vs direct reference (0 diffs, non-zero RGB); cpu median=${summary.cpu_ms_median.toFixed(3)}ms p95=${summary.cpu_ms_p95.toFixed(3)}ms; gpu median=${summary.gpu_elapsed_ms_median.toFixed(3)}ms p95=${summary.gpu_elapsed_ms_p95.toFixed(3)}ms; throughput fps median=${summary.throughput_fps_median.toFixed(2)}; init median=${initStr}; counts/frame: ${countsStr}; timing notes: cpu_ms runs from frame rows in to queue submitted; gpu_elapsed_ms runs from submit start to completion callback, including credit-loop wait; elapsed_ms is round wall time over measured frames only; total lane wall time=${totalWallTimeMs.toFixed(3)}ms`,
        implementation_owner: owner,
      };
    }
  }

  return { variants, measurements };
}

/**
 * Runs tmt.5 workload (a) in MEASUREMENT MODE (§6.4, §16.7, §21).
 */
export async function testWorkloadAMeasurementMode(bridge, wasmExports, externalFactories = null) {
  if (!bridge || !bridge.device) throw new Error("WebGpuBridgeHost device not initialized");

  const sharedRows = buildSharedRows();
  const repeatedFrames = buildRepeatedFrames(sharedRows, WORKLOAD_AC_MEASURE_TOTAL_FRAMES);

  const variantKeys = [
    "workload_a_direct",
    "workload_a_bulk",
    "workload_a_chatty",
    "workload_a_generated",
  ];

  const createRunnerFn = (vKey, b, w, s) => {
    if (externalFactories && typeof externalFactories[vKey] === "function") {
      return externalFactories[vKey](b, w, s);
    }
    return createWorkloadARunner(vKey, b, w, s);
  };

  return runWorkloadMeasurement({
    laneName: "workload_a_measure",
    workloadLabel: "(a)",
    variantKeys,
    directKey: "workload_a_direct",
    createRunnerFn,
    bridge,
    wasmExports,
    sharedRows,
    repeatedFrames,
  });
}

/**
 * Runs tmt.5 workload (c) in MEASUREMENT MODE (§6.4, §16.7, §21).
 */
export async function testWorkloadCMeasurementMode(bridge, wasmExports, externalFactories = null) {
  if (!bridge || !bridge.device) throw new Error("WebGpuBridgeHost device not initialized");

  const sharedRows = buildSharedRows();
  const repeatedFrames = buildRepeatedFrames(sharedRows, WORKLOAD_AC_MEASURE_TOTAL_FRAMES);

  const variantKeys = [
    "workload_c_direct_bundle",
    "workload_c_bulk_bundle",
    "workload_c_chatty_bundle",
    "workload_c_generated_bundle",
  ];

  const createRunnerFn = (vKey, b, w, s) => {
    if (externalFactories && typeof externalFactories[vKey] === "function") {
      return externalFactories[vKey](b, w, s);
    }
    return createWorkloadCRunner(vKey, b, w, s);
  };

  return runWorkloadMeasurement({
    laneName: "workload_c_measure",
    workloadLabel: "(c)",
    variantKeys,
    directKey: "workload_c_direct_bundle",
    createRunnerFn,
    bridge,
    wasmExports,
    sharedRows,
    repeatedFrames,
  });
}

/**
 * Runs tmt.5 borrowed-view measurement mode (§6.4, §6.6, §16.7, §21, tmt.5 STEP 3).
 *
 * Measures workload_a_direct, workload_a_bulk, and workload_a_bulk_borrowed
 * with the exact same rounds, warmup, init passes, and pixel identity.
 */
export async function testBorrowedBulkMeasurementMode(bridge, wasmExports, memory, externalFactories = null) {
  if (!bridge || !bridge.device) throw new Error("WebGpuBridgeHost device not initialized");

  const sharedRows = buildSharedRows();
  const repeatedFrames = buildRepeatedFrames(sharedRows, WORKLOAD_AC_MEASURE_TOTAL_FRAMES);

  const variantKeys = [
    "workload_a_direct",
    "workload_a_bulk",
    "workload_a_bulk_borrowed",
  ];

  const createRunnerFn = (vKey, b, w, s) => {
    if (externalFactories && typeof externalFactories[vKey] === "function") {
      return externalFactories[vKey](b, w, s, memory);
    }
    return createWorkloadARunner(vKey, b, w, s, memory);
  };

  return runWorkloadMeasurement({
    laneName: "borrowed_bulk_measure",
    workloadLabel: "(a)",
    variantKeys,
    directKey: "workload_a_direct",
    createRunnerFn,
    bridge,
    wasmExports,
    sharedRows,
    repeatedFrames,
  });
}

