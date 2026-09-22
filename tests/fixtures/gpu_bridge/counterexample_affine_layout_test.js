/**
 * tests/fixtures/gpu_bridge/counterexample_affine_layout_test.js
 *
 * Counterexample: AffineRows vs mat4x3 layout (§7.5, §11.7, vqa.7).
 *
 * Guards against silent 48/64-byte layout corruption between packed AffineRows
 * (3 rows of 4 floats, 48 bytes) and WGSL mat4x3<f32> (4 columns of 3 floats,
 * each column aligned to 16 bytes, requiring 64 bytes).
 *
 * Exercises:
 * - CobaltOrchid: Rust export f3d_build_affine_rows_layout_counterexample_packet(wrong_mat4x3_layout: bool)
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

const WIDTH = 64;
const HEIGHT = 64;
const BYTES_PER_ROW = 256;
const READBACK_SIZE = BYTES_PER_ROW * HEIGHT; // 16384 bytes

// Pixel offsets for (x, y) = (col, row):
// pixel (48, 32): 32 * 256 + 48 * 4 = 8384
const OFFSET_P48_32 = 32 * BYTES_PER_ROW + 48 * 4;
// pixel (32, 32): 32 * 256 + 32 * 4 = 8320
const OFFSET_P32_32 = 32 * BYTES_PER_ROW + 32 * 4;

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

export async function testAffineLayoutCounterexample(wasmExports) {
  const buildPacketFn =
    wasmExports.f3d_build_affine_rows_layout_counterexample_packet ||
    wasmExports.gpu_bridge_build_affine_rows_layout_counterexample_packet;

  if (typeof buildPacketFn !== "function") {
    throw new Error(
      "Missing required export 'f3d_build_affine_rows_layout_counterexample_packet' on wasmExports",
    );
  }

  // 1. CORRECT RUN (wrong_mat4x3_layout: false)
  // Translation +0.5 x shifts the green triangle so:
  // - pixel (48,32) @ 8384 is drawn: G>200, R<50, B<50, A=255
  // - pixel (32,32) @ 8320 is empty background: R<50, G<50, B<50, A=255
  const correctHost = new WebGpuBridgeHost();
  let correctP48, correctP32;

  try {
    await correctHost.negotiateAndCreateDevice({ requiredFeatures: [] });
    const packet = buildPacketFn(false);
    await correctHost.executePacket(packet);

    const pixels = await correctHost.readbackBuffer(20, READBACK_SIZE);
    correctP48 = [
      pixels[OFFSET_P48_32],
      pixels[OFFSET_P48_32 + 1],
      pixels[OFFSET_P48_32 + 2],
      pixels[OFFSET_P48_32 + 3],
    ];
    correctP32 = [
      pixels[OFFSET_P32_32],
      pixels[OFFSET_P32_32 + 1],
      pixels[OFFSET_P32_32 + 2],
      pixels[OFFSET_P32_32 + 3],
    ];
  } finally {
    cleanupHost(correctHost);
  }

  const correctP48Valid =
    correctP48[1] > 200 && correctP48[0] < 50 && correctP48[2] < 50 && correctP48[3] === 255;
  const correctP32Valid =
    correctP32[0] < 50 && correctP32[1] < 50 && correctP32[2] < 50 && correctP32[3] === 255;

  if (!correctP48Valid || !correctP32Valid) {
    throw new Error(
      `Correct run failed (wrong_mat4x3_layout=false): observed p(48,32)=[${correctP48.join(",")}], p(32,32)=[${correctP32.join(",")}]; expected p(48,32) G>200,R<50,B<50,A=255 and p(32,32) R,G,B<50,A=255`,
    );
  }

  // 2. WRONG-LAYOUT RUN (wrong_mat4x3_layout: true)
  // Runs only after the correct run passes, on a fresh host, same readback.
  // It must RESOLVE with no error (the corruption is silent).
  // Due to the 64-byte column layout reading 48-byte packed rows, columns are misaligned
  // and the translation column is corrupted/zeroed, leaving the geometry centered:
  // - pixel (32,32) @ 8320 is drawn: G>200, R<50, B<50, A=255
  // - pixel (48,32) @ 8384 is background: R<50, G<50, B<50, A=255
  // A rejection or blank output is a FAIL, not a detection.
  const wrongHost = new WebGpuBridgeHost();
  let wrongP48, wrongP32;

  try {
    await wrongHost.negotiateAndCreateDevice({ requiredFeatures: [] });
    const packet = buildPacketFn(true);
    await wrongHost.executePacket(packet);

    const pixels = await wrongHost.readbackBuffer(20, READBACK_SIZE);
    wrongP48 = [
      pixels[OFFSET_P48_32],
      pixels[OFFSET_P48_32 + 1],
      pixels[OFFSET_P48_32 + 2],
      pixels[OFFSET_P48_32 + 3],
    ];
    wrongP32 = [
      pixels[OFFSET_P32_32],
      pixels[OFFSET_P32_32 + 1],
      pixels[OFFSET_P32_32 + 2],
      pixels[OFFSET_P32_32 + 3],
    ];
  } finally {
    cleanupHost(wrongHost);
  }

  const wrongP32Valid =
    wrongP32[1] > 200 && wrongP32[0] < 50 && wrongP32[2] < 50 && wrongP32[3] === 255;
  const wrongP48Valid =
    wrongP48[0] < 50 && wrongP48[1] < 50 && wrongP48[2] < 50 && wrongP48[3] === 255;

  if (!wrongP32Valid || !wrongP48Valid) {
    throw new Error(
      `Wrong-layout run failed (wrong_mat4x3_layout=true): observed p(32,32)=[${wrongP32.join(",")}], p(48,32)=[${wrongP48.join(",")}]; expected p(32,32) G>200,R<50,B<50,A=255 and p(48,32) R,G,B<50,A=255 (silent corruption proof)`,
    );
  }

  const detail =
    `Correct (AffineRows): p(48,32)=[${correctP48.join(",")}], p(32,32)=[${correctP32.join(",")}]; ` +
    `Wrong (mat4x3 silent shift): p(32,32)=[${wrongP32.join(",")}], p(48,32)=[${wrongP48.join(",")}] ` +
    `(owner: new-backend).`;

  return {
    status: "PASS",
    observed_pixels: {
      correct: {
        p48_32: correctP48,
        p32_32: correctP32,
      },
      wrong_layout: {
        p32_32: wrongP32,
        p48_32: wrongP48,
      },
    },
    expected_ranges: {
      correct: "p(48,32) G>200, R<50, B<50, A=255; p(32,32) R<50, G<50, B<50, A=255",
      wrong_layout: "p(32,32) G>200, R<50, B<50, A=255; p(48,32) R<50, G<50, B<50, A=255",
    },
    detail,
    implementation_owner: "new-backend",
    owner: "new-backend",
    [false]: {
      status: "PASS",
      wrong_mat4x3_layout: false,
      observed_pixels: { p48_32: correctP48, p32_32: correctP32 },
      owner: "new-backend",
    },
    [true]: {
      status: "PASS",
      wrong_mat4x3_layout: true,
      observed_pixels: { p32_32: wrongP32, p48_32: wrongP48 },
      owner: "new-backend",
    },
  };
}
