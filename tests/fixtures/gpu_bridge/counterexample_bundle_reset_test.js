/**
 * tests/fixtures/gpu_bridge/counterexample_bundle_reset_test.js
 *
 * Counterexample: Bundle-then-direct render pass state reset (§8.2, vqa.7).
 *
 * Validates that render-pass state (pipeline, bind groups, vertex buffers) is invalidated
 * by executeBundles (both non-empty executeBundles([1]) and empty executeBundles([])),
 * guarding against a falsely warm binding/pipeline cache in the bridge host decoder.
 *
 * Exercises:
 * - CobaltOrchid: Rust export f3d_build_bundle_direct_warm_cache_packet(empty_bundle_list: bool)
 * - GrayFox: WebGpuBridgeHost({ wrongImplSkipBundleStateReset: true })
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

// Exact sample coordinates sourced from tests/fixtures/gpu_bridge/variant_preconditions.js:33-35 (lines 500-524)
const WIDTH = 64;
const HEIGHT = 64;
const BYTES_PER_ROW = 256;
const READBACK_SIZE = BYTES_PER_ROW * HEIGHT; // 16384 bytes
const LEFT_GREEN_OFFSET = 48 * BYTES_PER_ROW + 19 * 4; // 12364 (col 19, row 48) - variant_preconditions.js:33
const RIGHT_BLUE_OFFSET = 48 * BYTES_PER_ROW + 51 * 4; // 12492 (col 51, row 48) - variant_preconditions.js:34
const BACKGROUND_OFFSET = 10 * BYTES_PER_ROW + 10 * 4; // 2600 (col 10, row 10)  - variant_preconditions.js:35

function cleanupHost(host) {
  if (!host) return;
  if (host.buffers) {
    for (const b of host.buffers.values()) {
      try {
        b.destroy();
      } catch (_) {}
    }
  }
  if (host.textures) {
    for (const t of host.textures.values()) {
      try {
        t.destroy();
      } catch (_) {}
    }
  }
  try {
    host.destroyDevice();
  } catch (_) {}
}

export async function testBundleResetCounterexample(wasmExports) {
  const buildPacketFn =
    wasmExports.f3d_build_bundle_direct_warm_cache_packet ||
    wasmExports.gpu_bridge_build_bundle_direct_warm_cache_packet;

  if (typeof buildPacketFn !== "function") {
    throw new Error(
      "Missing required export 'f3d_build_bundle_direct_warm_cache_packet' on wasmExports",
    );
  }

  const results = {};

  for (const emptyBundleList of [false, true]) {
    // 1. CORRECT RUN
    const host = new WebGpuBridgeHost();
    let leftPixel, rightPixel, bgPixel;

    try {
      await host.negotiateAndCreateDevice({ requiredFeatures: [] });
      const packet = buildPacketFn(emptyBundleList);
      await host.executePacket(packet);

      const pixels = await host.readbackBuffer(20, READBACK_SIZE);
      leftPixel = [
        pixels[LEFT_GREEN_OFFSET],
        pixels[LEFT_GREEN_OFFSET + 1],
        pixels[LEFT_GREEN_OFFSET + 2],
        pixels[LEFT_GREEN_OFFSET + 3],
      ];
      rightPixel = [
        pixels[RIGHT_BLUE_OFFSET],
        pixels[RIGHT_BLUE_OFFSET + 1],
        pixels[RIGHT_BLUE_OFFSET + 2],
        pixels[RIGHT_BLUE_OFFSET + 3],
      ];
      bgPixel = [
        pixels[BACKGROUND_OFFSET],
        pixels[BACKGROUND_OFFSET + 1],
        pixels[BACKGROUND_OFFSET + 2],
        pixels[BACKGROUND_OFFSET + 3],
      ];
    } finally {
      cleanupHost(host);
    }

    // Validation of pixel ranges for the CORRECT run:
    let expectedRanges;
    if (!emptyBundleList) {
      // false: left G>200, R<50, B<50; right B>200, R<50, G<50; background R,G,B<50; alpha 255 on all three.
      expectedRanges =
        "left G>200, R<50, B<50, A=255; right B>200, R<50, G<50, A=255; bg R<50, G<50, B<50, A=255";
      const leftValid =
        leftPixel[1] > 200 && leftPixel[0] < 50 && leftPixel[2] < 50 && leftPixel[3] === 255;
      const rightValid =
        rightPixel[2] > 200 && rightPixel[0] < 50 && rightPixel[1] < 50 && rightPixel[3] === 255;
      const bgValid = bgPixel[0] < 50 && bgPixel[1] < 50 && bgPixel[2] < 50 && bgPixel[3] === 255;

      if (!leftValid || !rightValid || !bgValid) {
        throw new Error(
          `Correct run failed for emptyBundleList=false: observed left=[${leftPixel}], right=[${rightPixel}], bg=[${bgPixel}]; expected: ${expectedRanges}`,
        );
      }
    } else {
      // true: right blue as above; left sample R,G,B<50 with alpha 255; background black with alpha 255.
      expectedRanges =
        "left R<50, G<50, B<50, A=255; right B>200, R<50, G<50, A=255; bg R<50, G<50, B<50, A=255";
      const rightValid =
        rightPixel[2] > 200 && rightPixel[0] < 50 && rightPixel[1] < 50 && rightPixel[3] === 255;
      const leftValid =
        leftPixel[0] < 50 && leftPixel[1] < 50 && leftPixel[2] < 50 && leftPixel[3] === 255;
      const bgValid = bgPixel[0] < 50 && bgPixel[1] < 50 && bgPixel[2] < 50 && bgPixel[3] === 255;

      if (!leftValid || !rightValid || !bgValid) {
        throw new Error(
          `Correct run failed for emptyBundleList=true: observed left=[${leftPixel}], right=[${rightPixel}], bg=[${bgPixel}]; expected: ${expectedRanges}`,
        );
      }
    }

    // 2. WRONG-IMPL RUN: run only after CORRECT run passes
    const wrongHost = new WebGpuBridgeHost({ wrongImplSkipBundleStateReset: true });
    let wrongError = null;
    let wrongPixels = null;

    try {
      await wrongHost.negotiateAndCreateDevice({ requiredFeatures: [] });
      const packet = buildPacketFn(emptyBundleList);
      await wrongHost.executePacket(packet);

      // If it erroneously resolves, read back pixels to report failure details
      try {
        wrongPixels = await wrongHost.readbackBuffer(20, READBACK_SIZE);
      } catch (_) {}
    } catch (err) {
      wrongError = err;
    } finally {
      cleanupHost(wrongHost);
    }

    if (!wrongError) {
      const pSample = wrongPixels
        ? `sampled=[${wrongPixels[LEFT_GREEN_OFFSET]},${wrongPixels[RIGHT_BLUE_OFFSET]},${wrongPixels[BACKGROUND_OFFSET]}]`
        : "no readback";
      throw new Error(
        `Wrong-impl run falsely resolved for emptyBundleList=${emptyBundleList}. Expected WebGPU error scope rejection, but execution succeeded (${pSample}).`,
      );
    }

    const wrongErrorMsg = wrongError.message || String(wrongError);
    if (!wrongErrorMsg.startsWith("WebGPU error scope reported error:")) {
      throw new Error(
        `Wrong-impl run rejected with unexpected error (expected prefix 'WebGPU error scope reported error:'): "${wrongErrorMsg}"`,
      );
    }

    const observedPixels = {
      left: leftPixel,
      right: rightPixel,
      background: bgPixel,
    };

    const runDetail =
      `emptyBundleList=${emptyBundleList}: Correct observed left=[${leftPixel.join(",")}], right=[${rightPixel.join(",")}], bg=[${bgPixel.join(",")}]; ` +
      `Wrong-impl rejected as expected: "${wrongErrorMsg}" (owner: new-backend).`;

    const entry = {
      status: "PASS",
      empty_bundle_list: emptyBundleList,
      observed_pixels: observedPixels,
      expected_ranges: expectedRanges,
      wrong_impl_error: wrongErrorMsg,
      detail: runDetail,
      implementation_owner: "new-backend",
      owner: "new-backend",
    };

    results[emptyBundleList] = entry;
    if (!emptyBundleList) {
      results.counterexample_bundle_then_direct_state_reset = entry;
    } else {
      results.counterexample_empty_bundle_list_state_reset = entry;
    }
  }

  return results;
}
