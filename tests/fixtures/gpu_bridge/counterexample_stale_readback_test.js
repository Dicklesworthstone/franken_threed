/**
 * tests/fixtures/gpu_bridge/counterexample_stale_readback_test.js
 *
 * Counterexample: Stale publication into a replaced device (§6.5, §11.7, vqa.7).
 *
 * Validates that an in-flight asynchronous readback (mapAsync) initiated on an
 * older device generation is rejected and never publishes stale GPU bytes into a
 * replaced device session when the device is replaced while mapping is pending.
 *
 * Exercises:
 * - GrayFox: host.installDeviceForTest(adapter, device)
 * - GrayFox: WebGpuBridgeHost({ wrongImplSkipReadbackDeviceCheck: true })
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

const BYTES_PER_ROW = 256;
const HEIGHT = 64;
const READBACK_SIZE = BYTES_PER_ROW * HEIGHT; // 16384 bytes
const CENTER_OFFSET = 32 * BYTES_PER_ROW + 32 * 4; // 8320

export async function testStaleReadbackCounterexample(wasmExports) {
  const buildPacketFn =
    wasmExports.f3d_build_red_a_blue_b_packet || wasmExports.gpu_bridge_build_red_blue_packet;

  if (typeof buildPacketFn !== "function") {
    throw new Error("Missing required export 'f3d_build_red_a_blue_b_packet' on wasmExports");
  }

  const results = {};

  for (const wrongImpl of [false, true]) {
    // 1. Initialize host with the designated check flag
    const host = new WebGpuBridgeHost({
      wrongImplSkipReadbackDeviceCheck: wrongImpl,
    });
    await host.negotiateAndCreateDevice({ requiredFeatures: [] });

    // 2. Create the replacement BEFORE the readback
    const adapterB = await navigator.gpu.requestAdapter();
    const deviceB = await adapterB.requestDevice();

    let oldDevice = null;
    let oldGen = null;
    let outcome = null;

    try {
      // 3. Render red-A/blue-B packet. Buffer 40 holds red.
      await host.executePacket(buildPacketFn(true));

      // 4. Capture current device and generation
      oldDevice = host.device;
      oldGen = host.deviceGeneration;

      // 5. Start readback buffer 40 asynchronously
      const pending = host.readbackBuffer(40, READBACK_SIZE).then(
        (v) => ({ published: true, v }),
        (e) => ({ published: false, e }),
      );

      // 6. In the same synchronous task, replace device immediately
      host.installDeviceForTest(adapterB, deviceB);
      if (host.device !== deviceB) {
        throw new Error("installDeviceForTest failed: host.device does not match deviceB");
      }
      if (host.deviceGeneration !== oldGen + 1) {
        throw new Error(
          `installDeviceForTest failed: host.deviceGeneration is ${host.deviceGeneration}, expected ${oldGen + 1}`,
        );
      }

      // 7. Await outcome of pending readback
      outcome = await pending;
    } finally {
      // 8. Clean up all devices
      try {
        host.destroyDevice();
      } catch (_) {}
      if (oldDevice) {
        try {
          oldDevice.destroy();
        } catch (_) {}
      }
      if (deviceB) {
        try {
          deviceB.destroy();
        } catch (_) {}
      }
    }

    if (!wrongImpl) {
      // wrongImpl=false: require outcome.published === false AND message to include
      // "device changed before mapping completed". Any other rejection or a publish is a FAIL.
      if (outcome.published !== false) {
        throw new Error(
          "Correct run (wrongImpl=false) unexpectedly published readback across device replacement",
        );
      }
      const errorMsg = (outcome.e && (outcome.e.message || String(outcome.e))) || "";
      if (!errorMsg.includes("device changed before mapping completed")) {
        throw new Error(
          `Correct run (wrongImpl=false) rejected with unexpected error: "${errorMsg}" (expected to include "device changed before mapping completed")`,
        );
      }

      results[false] = {
        status: "PASS",
        wrong_impl: false,
        published: false,
        error_message: errorMsg,
        old_generation: oldGen,
        new_generation: oldGen + 1,
        owner: "new-backend",
      };
    } else {
      // wrongImpl=true: require outcome.published === true;
      // center offset 32*256+32*4: R>200, B<50, A=255 (real old-device GPU bytes);
      // and outcome.v.deviceGeneration === oldGen, which is stale against current oldGen + 1.
      // Anything else is a FAIL.
      if (outcome.published !== true) {
        const err = outcome.e && (outcome.e.message || String(outcome.e));
        throw new Error(`Wrong-impl run (wrongImpl=true) unexpectedly failed to publish: ${err}`);
      }

      const pixel = [
        outcome.v[CENTER_OFFSET],
        outcome.v[CENTER_OFFSET + 1],
        outcome.v[CENTER_OFFSET + 2],
        outcome.v[CENTER_OFFSET + 3],
      ];
      const isRed = pixel[0] > 200 && pixel[2] < 50 && pixel[3] === 255;

      if (!isRed) {
        throw new Error(
          `Wrong-impl run (wrongImpl=true) published corrupted/blank bytes: pixel=[${pixel.join(",")}], expected R>200, B<50, A=255`,
        );
      }

      if (outcome.v.deviceGeneration !== oldGen) {
        throw new Error(
          `Wrong-impl run (wrongImpl=true) outcome generation is ${outcome.v.deviceGeneration}, expected stale generation ${oldGen}`,
        );
      }

      results[true] = {
        status: "PASS",
        wrong_impl: true,
        published: true,
        observed_pixel: pixel,
        reported_generation: outcome.v.deviceGeneration,
        current_generation: oldGen + 1,
        owner: "new-backend",
      };
    }
  }

  const detail =
    `Correct (guard active): rejected publication with "${results[false].error_message}" (gen ${results[false].old_generation} -> ${results[false].new_generation}); ` +
    `Wrong-impl (guard skipped): falsely published stale bytes (center pixel=[${results[true].observed_pixel.join(",")}], stale gen ${results[true].reported_generation} vs current ${results[true].current_generation}) ` +
    `(owner: new-backend).`;

  results.status = "PASS";
  results.detail = detail;
  results.implementation_owner = "new-backend";
  results.owner = "new-backend";

  return results;
}
