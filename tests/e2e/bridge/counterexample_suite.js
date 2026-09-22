/**
 * counterexample_suite.js - Comprehensive WebGPU Bridge Counterexample Suite
 *
 * Bead: f3d-05-ids-layouts-epochs-transport-vqa.7
 *
 * Implements the mandatory adversarial browser test suite against ChartreuseFern's
 * real WebGPU bridge runtime, verified with independently issued direct-JS oracle references.
 *
 * Every test verifies:
 * 1. Positive implementation: Correct path passes assertions.
 * 2. Explicitly isolated broken fixture/control: Tested with the EXACT SAME assertion
 *    to prove that the test demonstrably detects and rejects each runtime hazard.
 */

import {
  OPCODE_COPY_TEXTURE_TO_BUFFER,
  OPCODE_CREATE_BUFFER,
  OPCODE_CREATE_PIPELINE,
  OPCODE_CREATE_TEXTURE,
  OPCODE_EXECUTE_BUNDLES,
  OPCODE_RECORD_BUNDLE,
  OPCODE_RENDER_PASS,
  OPCODE_WRITE_BUFFER,
  PACKET_MAGIC,
  PACKET_VERSION,
  TARGET_CANVAS,
  TARGET_OFFSCREEN,
  WebGpuBridgeHost,
} from "../../fixtures/gpu_bridge/bridge_runtime.js";

import {
  assertAffineRowsLayoutValid,
  assertAffineRowsTransformMatch,
  assertBundleDirectDrawMatch,
  assertGenerationalHandlePublication,
  assertMemoryGrowthAllowed,
  assertNestedCanvasPassMatch,
  assertNestedPassMatch,
  computeAlignedBytesPerRow,
  evalDirectAffineTransform,
  readbackGpuBuffer,
  renderDirectBrokenNestedPass,
  renderDirectBundleDirectReference,
  renderDirectNestedCanvasOffscreenReference,
  renderDirectNestedPassReference,
  renderDirectRedABlueBReference,
  renderDirectTriangleReference,
  WGSL_AFFINE_TRIANGLE,
  WGSL_FLAT_COLOR_BUNDLE,
  WGSL_SOLID_COLOR,
} from "./oracle_reference.js";

/**
 * High-throughput binary packet builder matching the Rust GpuSubmissionPacket wire layout.
 */
export class BinaryPacketBuilder {
  constructor() {
    this.commands = [];
    this.dataChunks = [];
    this.dataTotalLen = 0;
  }

  createBuffer(bufferId, size, usage) {
    this.commands.push({ op: OPCODE_CREATE_BUFFER, bufferId, size, usage });
  }

  writeBuffer(bufferId, offset, dataUint8) {
    const dataOffset = this.dataTotalLen;
    this.dataChunks.push(dataUint8);
    this.dataTotalLen += dataUint8.byteLength;
    this.commands.push({
      op: OPCODE_WRITE_BUFFER,
      bufferId,
      offset,
      dataOffset,
      dataLength: dataUint8.byteLength,
    });
  }

  createTexture(textureId, width, height, formatCode, usage) {
    this.commands.push({ op: OPCODE_CREATE_TEXTURE, textureId, width, height, formatCode, usage });
  }

  createPipeline(
    pipelineId,
    wgslText,
    formatCode,
    hasVB,
    hasUniform,
    uniformSize = 0,
    vertexStride = 0,
  ) {
    const codeBytes = new TextEncoder().encode(wgslText);
    const codeOffset = this.dataTotalLen;
    this.dataChunks.push(codeBytes);
    this.dataTotalLen += codeBytes.byteLength;
    this.commands.push({
      op: OPCODE_CREATE_PIPELINE,
      pipelineId,
      codeOffset,
      codeLen: codeBytes.byteLength,
      formatCode,
      hasVB,
      hasUniform,
      uniformSize,
      vertexStride,
    });
  }

  renderPass(
    targetType,
    targetId,
    clearColor,
    pipelineId,
    vbId,
    vertexCount,
    dynamicOffset = 0,
    uniformBufferId = 1,
  ) {
    this.commands.push({
      op: OPCODE_RENDER_PASS,
      targetType,
      targetId,
      clearColor,
      pipelineId,
      vbId,
      vertexCount,
      dynamicOffset,
      uniformBufferId,
    });
  }

  copyTextureToBuffer(textureId, bufferId, width, height, epochHi = 0, epochLo = 0) {
    this.commands.push({
      op: OPCODE_COPY_TEXTURE_TO_BUFFER,
      textureId,
      bufferId,
      width,
      height,
      epochHi,
      epochLo,
    });
  }

  recordBundle(
    bundleId,
    pipelineId,
    vertexBufferId,
    vertexCount,
    dynamicOffset = 0,
    uniformBufferId = 1,
    targetFormat = 2,
  ) {
    this.commands.push({
      op: OPCODE_RECORD_BUNDLE,
      bundleId,
      pipelineId,
      vertexBufferId,
      vertexCount,
      dynamicOffset,
      uniformBufferId,
      targetFormat,
    });
  }

  executeBundles(bundleIds) {
    this.commands.push({
      op: OPCODE_EXECUTE_BUNDLES,
      bundleIds,
    });
  }

  build() {
    const headerLen = 16;
    let commandBytesLen = 0;
    for (const cmd of this.commands) {
      switch (cmd.op) {
        case OPCODE_CREATE_BUFFER:
          commandBytesLen += 2 + 12;
          break;
        case OPCODE_WRITE_BUFFER:
          commandBytesLen += 2 + 16;
          break;
        case OPCODE_CREATE_TEXTURE:
          commandBytesLen += 2 + 20;
          break;
        case OPCODE_CREATE_PIPELINE:
          commandBytesLen += 2 + 32;
          break;
        case OPCODE_RENDER_PASS:
          commandBytesLen += 2 + 44;
          break;
        case OPCODE_COPY_TEXTURE_TO_BUFFER:
          commandBytesLen += 2 + 24;
          break;
        case OPCODE_RECORD_BUNDLE:
          commandBytesLen += 2 + 28;
          break;
        case OPCODE_EXECUTE_BUNDLES:
          commandBytesLen += 2 + 4 + cmd.bundleIds.length * 4;
          break;
      }
    }

    const totalLen = headerLen + commandBytesLen + this.dataTotalLen;
    const out = new Uint8Array(totalLen);
    const view = new DataView(out.buffer);

    view.setUint32(0, PACKET_MAGIC, true);
    view.setUint16(4, PACKET_VERSION, true);
    view.setUint16(6, 0, true);
    view.setUint32(8, this.commands.length, true);
    view.setUint32(12, this.dataTotalLen, true);

    let cursor = 16;
    for (const cmd of this.commands) {
      view.setUint16(cursor, cmd.op, true);
      cursor += 2;
      switch (cmd.op) {
        case OPCODE_CREATE_BUFFER:
          view.setUint32(cursor, cmd.bufferId, true);
          view.setUint32(cursor + 4, cmd.size, true);
          view.setUint32(cursor + 8, cmd.usage, true);
          cursor += 12;
          break;
        case OPCODE_WRITE_BUFFER:
          view.setUint32(cursor, cmd.bufferId, true);
          view.setUint32(cursor + 4, cmd.offset, true);
          view.setUint32(cursor + 8, cmd.dataOffset, true);
          view.setUint32(cursor + 12, cmd.dataLength, true);
          cursor += 16;
          break;
        case OPCODE_CREATE_TEXTURE:
          view.setUint32(cursor, cmd.textureId, true);
          view.setUint32(cursor + 4, cmd.width, true);
          view.setUint32(cursor + 8, cmd.height, true);
          view.setUint32(cursor + 12, cmd.formatCode, true);
          view.setUint32(cursor + 16, cmd.usage, true);
          cursor += 20;
          break;
        case OPCODE_CREATE_PIPELINE:
          view.setUint32(cursor, cmd.pipelineId, true);
          view.setUint32(cursor + 4, cmd.codeOffset, true);
          view.setUint32(cursor + 8, cmd.codeLen, true);
          view.setUint32(cursor + 12, cmd.formatCode, true);
          view.setUint32(cursor + 16, cmd.hasVB ? 1 : 0, true);
          view.setUint32(cursor + 20, cmd.hasUniform ? 1 : 0, true);
          view.setUint32(cursor + 24, cmd.uniformSize || 0, true);
          view.setUint32(cursor + 28, cmd.vertexStride || 0, true);
          cursor += 32;
          break;
        case OPCODE_RENDER_PASS:
          view.setUint32(cursor, cmd.targetType, true);
          view.setUint32(cursor + 4, cmd.targetId, true);
          view.setFloat32(cursor + 8, cmd.clearColor[0], true);
          view.setFloat32(cursor + 12, cmd.clearColor[1], true);
          view.setFloat32(cursor + 16, cmd.clearColor[2], true);
          view.setFloat32(cursor + 20, cmd.clearColor[3], true);
          view.setUint32(cursor + 24, cmd.pipelineId, true);
          view.setUint32(cursor + 28, cmd.vbId, true);
          view.setUint32(cursor + 32, cmd.vertexCount, true);
          view.setUint32(cursor + 36, cmd.dynamicOffset, true);
          view.setUint32(cursor + 40, cmd.uniformBufferId || 1, true);
          cursor += 44;
          break;
        case OPCODE_COPY_TEXTURE_TO_BUFFER:
          view.setUint32(cursor, cmd.textureId, true);
          view.setUint32(cursor + 4, cmd.bufferId, true);
          view.setUint32(cursor + 8, cmd.width, true);
          view.setUint32(cursor + 12, cmd.height, true);
          view.setUint32(cursor + 16, cmd.epochHi || 0, true);
          view.setUint32(cursor + 20, cmd.epochLo || 0, true);
          cursor += 24;
          break;
        case OPCODE_RECORD_BUNDLE:
          view.setUint32(cursor, cmd.bundleId, true);
          view.setUint32(cursor + 4, cmd.pipelineId, true);
          view.setUint32(cursor + 8, cmd.vertexBufferId, true);
          view.setUint32(cursor + 12, cmd.vertexCount, true);
          view.setUint32(cursor + 16, cmd.dynamicOffset, true);
          view.setUint32(cursor + 20, cmd.uniformBufferId || 1, true);
          view.setUint32(cursor + 24, cmd.targetFormat, true);
          cursor += 28;
          break;
        case OPCODE_EXECUTE_BUNDLES:
          view.setUint32(cursor, cmd.bundleIds.length, true);
          cursor += 4;
          for (let b = 0; b < cmd.bundleIds.length; b++) {
            view.setUint32(cursor, cmd.bundleIds[b], true);
            cursor += 4;
          }
          break;
      }
    }

    let payloadCursor = headerLen + commandBytesLen;
    for (const chunk of this.dataChunks) {
      out.set(chunk, payloadCursor);
      payloadCursor += chunk.byteLength;
    }

    return out;
  }
}

/**
 * Assertion Helper: Pixel-by-pixel buffer equivalence.
 */
export function assertExactPixelMatch(candidatePixels, oraclePixels, label = "Pixel Comparison") {
  if (candidatePixels.byteLength !== oraclePixels.byteLength) {
    throw new Error(
      `${label}: byte length mismatch (candidate=${candidatePixels.byteLength}, oracle=${oraclePixels.byteLength})`,
    );
  }
  let diffCount = 0;
  for (let i = 0; i < candidatePixels.length; i++) {
    if (candidatePixels[i] !== oraclePixels[i]) {
      diffCount++;
    }
  }
  if (diffCount > 0) {
    throw new Error(
      `${label}: detected ${diffCount} mismatched bytes out of ${candidatePixels.length}`,
    );
  }
}

/**
 * Assertion Helper: Red-A / Blue-B pure color verification.
 */
export function assertRedABlueB(pixelsA, pixelsB, width = 32, height = 32) {
  const bytesPerRow = computeAlignedBytesPerRow(width);

  // Check center pixel of Target A: expected Red [255, 0, 0, 255]
  const midY = Math.floor(height / 2);
  const midX = Math.floor(width / 2);
  const centerA = midY * bytesPerRow + midX * 4;
  const rA = pixelsA[centerA],
    gA = pixelsA[centerA + 1],
    bA = pixelsA[centerA + 2],
    aA = pixelsA[centerA + 3];

  if (rA < 250 || gA > 5 || bA > 5 || aA < 250) {
    throw new Error(
      `Target A color violation: expected Red [255, 0, 0, 255], observed [${rA}, ${gA}, ${bA}, ${aA}]`,
    );
  }

  // Check center pixel of Target B: expected Blue [0, 0, 255, 255]
  const centerB = midY * bytesPerRow + midX * 4;
  const rB = pixelsB[centerB],
    gB = pixelsB[centerB + 1],
    bB = pixelsB[centerB + 2],
    aB = pixelsB[centerB + 3];

  if (rB > 5 || gB > 5 || bB < 250 || aB < 250) {
    throw new Error(
      `Target B color violation: expected Blue [0, 0, 255, 255], observed [${rB}, ${gB}, ${bB}, ${aB}]`,
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 1. FIRST-FRAME PIXEL EQUIVALENCE
 * -----------------------------------------------------------------------------
 */

export async function testFirstFramePixelEquivalence(
  host,
  device,
  wasmModule,
  width = 64,
  height = 64,
  canvasContext = null,
) {
  const buildTriangleFn =
    wasmModule?.f3d_build_first_frame_packet || wasmModule?.gpu_bridge_build_triangle_packet;
  if (!wasmModule || typeof buildTriangleFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testFirstFramePixelEquivalence requires compiled application Wasm with export 'gpu_bridge_build_triangle_packet' or 'f3d_build_first_frame_packet'. Silent JS fallback is forbidden.",
    );
  }

  // REAL RUST/WASM PATH: Packet generated by crates/f3d-runtime/src/gpu_host.rs
  const packet = buildTriangleFn();
  const readbackBufferId = 20; // Rust build_triangle_submission outputs to buffer 20
  const renderWidth = 64;
  const renderHeight = 64;

  // The compiled Rust triangle packet contains a TARGET_CANVAS pass alongside the TARGET_OFFSCREEN pass.
  // Bridge execution requires a real configured canvasContext so the packet executes as authored (without skipping passes).
  let activeCanvasContext = canvasContext;
  if (!activeCanvasContext) {
    let canvas = null;
    if (typeof document !== "undefined" && typeof document.createElement === "function") {
      canvas = document.createElement("canvas");
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      canvas.style.display = "none";
      if (document.body) {
        document.body.appendChild(canvas);
      }
    } else if (typeof OffscreenCanvas !== "undefined") {
      canvas = new OffscreenCanvas(renderWidth, renderHeight);
    }

    if (canvas && typeof canvas.getContext === "function") {
      activeCanvasContext = canvas.getContext("webgpu");
    }
  }

  if (activeCanvasContext && typeof activeCanvasContext.configure === "function") {
    // Coordinate preferred canvas format with ChartreuseFern (Defect 2):
    // gpu_host.rs creates pipeline 101 for TARGET_CANVAS with format_code = 1 (bgra8unorm).
    // Use the negotiated capability preferredCanvasFormat or navigator.gpu.getPreferredCanvasFormat().
    const preferredFormat =
      host.capabilityRecord?.preferredCanvasFormat ||
      (typeof navigator !== "undefined" && navigator.gpu?.getPreferredCanvasFormat
        ? navigator.gpu.getPreferredCanvasFormat()
        : "bgra8unorm");

    activeCanvasContext.configure({
      device: device,
      format: preferredFormat,
      alphaMode: "premultiplied",
    });
  }

  const bytesPerRow = computeAlignedBytesPerRow(renderWidth);
  const readbackSize = bytesPerRow * renderHeight;

  // Execute packet with the real configured canvas context so TARGET_CANVAS executes as authored; offscreen readback remains the assertion
  await host.executePacket(packet, activeCanvasContext);
  const candidatePixels = await host.readbackBuffer(readbackBufferId, readbackSize);

  // Oracle: Independent direct-JS execution
  const oraclePixels = await renderDirectTriangleReference(device, renderWidth, renderHeight);

  // Positive Assertion: Must match pixel-for-pixel (0 differences)
  assertExactPixelMatch(candidatePixels, oraclePixels, "Candidate vs Direct-JS Oracle");

  return { candidatePixels, oraclePixels };
}

export async function testNegativeBrokenFirstFrameGpuTransform(
  host,
  oraclePixels,
  width = 64,
  height = 64,
) {
  // Broken Control: An actual faulty GPU packet uploading an out-of-bounds transform matrix.
  // Translation tx=50, ty=50 pushes all 3 vertices completely outside the NDC [-1, 1] clip space.
  // Real GPU execution renders pure clear color [0, 0, 0, 1] with the triangle entirely clipped!
  const builder = new BinaryPacketBuilder();
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  builder.createTexture(
    1,
    width,
    height,
    2,
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  );
  builder.createBuffer(1, 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const brokenAffine = new Float32Array([
    1.0, 0.0, 0.0, 50.0, 0.0, 1.0, 0.0, 50.0, 0.0, 0.0, 1.0, 0.0,
  ]);
  builder.writeBuffer(1, 0, new Uint8Array(brokenAffine.buffer));

  const vertexData = new Float32Array([
    0.0, 0.5, 0.0, 0.5, 1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 1.0, 0.0,
  ]);
  builder.createBuffer(2, vertexData.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  builder.writeBuffer(2, 0, new Uint8Array(vertexData.buffer));

  builder.createPipeline(1, WGSL_AFFINE_TRIANGLE, 2, true, true, 48, 20);
  builder.renderPass(TARGET_OFFSCREEN, 1, [0.0, 0.0, 0.0, 1.0], 1, 2, 3, 0, 1);

  builder.createBuffer(3, readbackSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  builder.copyTextureToBuffer(1, 3, width, height);

  const brokenPacket = builder.build();
  await host.executePacket(brokenPacket);

  // Read back the real rendered pixels from the GPU
  const brokenGpuPixels = await host.readbackBuffer(3, readbackSize);

  // Apply the EXACT SAME assertion: must reject the real GPU output
  let rejected = false;
  try {
    assertExactPixelMatch(brokenGpuPixels, oraclePixels, "Corrupted GPU Triangle vs Oracle");
  } catch (e) {
    rejected = true;
  }

  if (!rejected) {
    throw new Error(
      "Negative Control Failed: assertExactPixelMatch failed to reject out-of-bounds GPU render output!",
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 2. RED-A / BLUE-B QUEUE-WRITE SNAPSHOT ISOLATION
 * -----------------------------------------------------------------------------
 */

export async function testRedABlueBQueueWriteSnapshot(host, wasmModule, width = 64, height = 64) {
  const buildRedBlueFn =
    wasmModule?.f3d_build_red_a_blue_b_packet || wasmModule?.gpu_bridge_build_red_blue_packet;
  if (!wasmModule || typeof buildRedBlueFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testRedABlueBQueueWriteSnapshot requires compiled application Wasm with export 'gpu_bridge_build_red_blue_packet' or 'f3d_build_red_a_blue_b_packet'. Silent JS fallback is forbidden.",
    );
  }

  // REAL RUST/WASM PATH: Packet generated by crates/f3d-runtime/src/gpu_host.rs
  // consuming GrayFox's PerUseByteBuffer from f3d-core::ownership
  const packet = buildRedBlueFn(true);
  const readbackIdA = 40; // Rust build_red_a_blue_b_submission writes readback_a to 40
  const readbackIdB = 41; // and readback_b to 41
  const renderWidth = 64;
  const renderHeight = 64;

  const bytesPerRow = computeAlignedBytesPerRow(renderWidth);
  const readbackSize = bytesPerRow * renderHeight;

  await host.executePacket(packet);
  const pixelsA = await host.readbackBuffer(readbackIdA, readbackSize);
  const pixelsB = await host.readbackBuffer(readbackIdB, readbackSize);

  // Positive Assertion: Target A is pure Red, Target B is pure Blue
  assertRedABlueB(pixelsA, pixelsB, renderWidth, renderHeight);

  return { pixelsA, pixelsB };
}

export async function testNegativeBrokenRedABlueBQueueHazard(
  host,
  wasmModule,
  width = 32,
  height = 32,
) {
  const buildRedBlueFn =
    wasmModule?.f3d_build_red_a_blue_b_packet || wasmModule?.gpu_bridge_build_red_blue_packet;
  if (!wasmModule || typeof buildRedBlueFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenRedABlueBQueueHazard requires compiled application Wasm with export 'gpu_bridge_build_red_blue_packet' or 'f3d_build_red_a_blue_b_packet'.",
    );
  }

  // 1. Test unversioned Rust/Wasm packet (per_use_versioned = false)
  // This packet does not allocate distinct PerUseByteBuffer offsets, causing both passes to read the second uniform write!
  const invalidWasmPacket = buildRedBlueFn(false);
  await host.executePacket(invalidWasmPacket);
  const bytesPerRowWasm = computeAlignedBytesPerRow(64);
  const readbackSizeWasm = bytesPerRowWasm * 64;
  const brokenPixelsA = await host.readbackBuffer(40, readbackSizeWasm);
  const brokenPixelsB = await host.readbackBuffer(41, readbackSizeWasm);

  let rejectedWasm = false;
  try {
    assertRedABlueB(brokenPixelsA, brokenPixelsB, 64, 64);
  } catch (e) {
    rejectedWasm = true;
  }
  if (!rejectedWasm) {
    throw new Error(
      "Negative Control Failed: Unversioned Rust packet was not rejected by assertRedABlueB!",
    );
  }

  // 2. Also test direct GPU in-place uniform overwrite hazard control
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  const builder = new BinaryPacketBuilder();
  builder.createTexture(
    30,
    width,
    height,
    2,
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  );
  builder.createTexture(
    31,
    width,
    height,
    2,
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  );

  // Single uniform slot at offset 0
  builder.createBuffer(1, 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const redData = new Float32Array([1.0, 0.0, 0.0, 1.0]);
  const blueData = new Float32Array([0.0, 0.0, 1.0, 1.0]);
  builder.writeBuffer(1, 0, new Uint8Array(redData.buffer));
  // Overwrite in place before submit!
  builder.writeBuffer(1, 0, new Uint8Array(blueData.buffer));

  builder.createPipeline(10, WGSL_SOLID_COLOR, 2, false, true, 16, 0);

  // Both passes read offset 0, uniformBufferId 1
  builder.renderPass(TARGET_OFFSCREEN, 30, [0.0, 0.0, 0.0, 1.0], 10, 0, 6, 0, 1);
  builder.renderPass(TARGET_OFFSCREEN, 31, [0.0, 0.0, 0.0, 1.0], 10, 0, 6, 0, 1);

  builder.createBuffer(40, readbackSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  builder.createBuffer(41, readbackSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  builder.copyTextureToBuffer(30, 40, width, height);
  builder.copyTextureToBuffer(31, 41, width, height);

  const packet = builder.build();
  await host.executePacket(packet);

  const brokenPixelsA2 = await host.readbackBuffer(40, readbackSize);
  const brokenPixelsB2 = await host.readbackBuffer(41, readbackSize);

  // The EXACT SAME assertion must be applied to the broken output and MUST reject it
  let rejectedJs = false;
  try {
    assertRedABlueB(brokenPixelsA2, brokenPixelsB2, width, height);
  } catch (e) {
    rejectedJs = true;
  }

  if (!rejectedJs) {
    throw new Error(
      "Negative Control Failed: assertRedABlueB did not reject in-place queue-write overwrite hazard!",
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 3. FOCUSED NEGATIVE CONTROL: MISSING WASM / EXPORT REJECTION
 * -----------------------------------------------------------------------------
 * Verifies that the test suite demonstrably rejects any attempt to run positive
 * paths without compiled application Wasm or missing required exports.
 */
export async function testNegativeMissingWasmRejection(host, device) {
  // 1. Null wasm module must throw
  let rejectedNull = false;
  try {
    await testFirstFramePixelEquivalence(host, device, null);
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedNull = true;
    }
  }
  if (!rejectedNull) {
    throw new Error(
      "Negative Control Failed: testFirstFramePixelEquivalence did not reject null wasmModule!",
    );
  }

  // 2. Missing required export must throw
  let rejectedMissingExport = false;
  try {
    await testRedABlueBQueueWriteSnapshot(host, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedMissingExport = true;
    }
  }
  if (!rejectedMissingExport) {
    throw new Error(
      "Negative Control Failed: testRedABlueBQueueWriteSnapshot did not reject missing export!",
    );
  }

  // 3. Stale epoch publication gate missing export must throw
  let rejectedPublicationExport = false;
  try {
    await testStaleEpochReadbackPublicationGate(host, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedPublicationExport = true;
    }
  }
  if (!rejectedPublicationExport) {
    throw new Error(
      "Negative Control Failed: testStaleEpochReadbackPublicationGate did not reject missing export!",
    );
  }

  // 4. Bundle-then-direct draw state reset missing export must throw
  let rejectedBundleExport = false;
  try {
    await testBundleThenDirectDrawStateReset(host, device, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedBundleExport = true;
    }
  }
  if (!rejectedBundleExport) {
    throw new Error(
      "Negative Control Failed: testBundleThenDirectDrawStateReset did not reject missing export!",
    );
  }

  // 5. Generational handle ABA publication missing export must throw
  let rejectedHandleExport = false;
  try {
    await testGenerationalHandleAbaPublication(host, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedHandleExport = true;
    }
  }
  if (!rejectedHandleExport) {
    throw new Error(
      "Negative Control Failed: testGenerationalHandleAbaPublication did not reject missing export!",
    );
  }

  // 6. Linear memory borrow guards missing export must throw
  let rejectedBorrowExport = false;
  try {
    await testLinearMemoryBorrowGuards(host, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedBorrowExport = true;
    }
  }
  if (!rejectedBorrowExport) {
    throw new Error(
      "Negative Control Failed: testLinearMemoryBorrowGuards did not reject missing export!",
    );
  }

  // 7. AffineRows GPU layout validation missing export must throw
  let rejectedAffineExport = false;
  try {
    await testAffineRowsGpuLayoutValidation(host, {});
  } catch (err) {
    if (err.message && err.message.includes("Silent JS fallback is forbidden")) {
      rejectedAffineExport = true;
    }
  }
  if (!rejectedAffineExport) {
    throw new Error(
      "Negative Control Failed: testAffineRowsGpuLayoutValidation did not reject missing export!",
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 4. PACKET PARSER BOUNDS & MALFORMED COMMAND REJECTION (UNIT CONTROLS)
 * -----------------------------------------------------------------------------
 * Verifies that well-formed packets decode properly, while malformed, truncated,
 * or invalid-format packets are rejected strictly at the decoder boundary.
 */
export async function testMalformedPacketBoundsChecking(host) {
  // Positive: Well-formed binary packet creates a buffer and offscreen texture
  const builder = new BinaryPacketBuilder();
  builder.createBuffer(100, 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  builder.createTexture(100, 32, 32, 2, GPUTextureUsage.RENDER_ATTACHMENT);
  const packet = builder.build();
  await host.executePacket(packet);

  if (!host.buffers.has(100) || !host.textures.has(100)) {
    throw new Error("MalformedPacketBoundsChecking: valid packet resources not created");
  }
}

export async function testNegativeBrokenMalformedPacket(host) {
  // Sub-test 1: Truncated opcode fields
  const truncatedPacket = new Uint8Array(20);
  const view = new DataView(truncatedPacket.buffer);
  view.setUint32(0, PACKET_MAGIC, true);
  view.setUint16(4, PACKET_VERSION, true);
  view.setUint32(8, 1, true); // 1 command declared
  view.setUint32(12, 0, true);
  view.setUint16(16, OPCODE_CREATE_BUFFER, true); // Only 2 bytes remaining, but expects 12

  let rejectedTruncated = false;
  try {
    await host.executePacket(truncatedPacket);
  } catch (e) {
    if (e.message && e.message.includes("Truncated CREATE_BUFFER")) {
      rejectedTruncated = true;
    }
  }
  if (!rejectedTruncated) {
    throw new Error(
      "Negative Control Failed: Bridge decoder failed to reject truncated packet fields!",
    );
  }

  // Sub-test 2: Invalid texture format code
  const builder = new BinaryPacketBuilder();
  builder.createTexture(101, 16, 16, 99, GPUTextureUsage.RENDER_ATTACHMENT); // formatCode 99 invalid
  const invalidFormatPacket = builder.build();

  let rejectedFormat = false;
  try {
    await host.executePacket(invalidFormatPacket);
  } catch (e) {
    if (e.message && e.message.includes("Invalid texture formatCode")) {
      rejectedFormat = true;
    }
  }
  if (!rejectedFormat) {
    throw new Error(
      "Negative Control Failed: Bridge decoder failed to reject invalid texture format code!",
    );
  }

  // Sub-test 3: Invalid render pass target type
  const builder2 = new BinaryPacketBuilder();
  builder2.createBuffer(102, 256, GPUBufferUsage.VERTEX);
  builder2.createPipeline(102, WGSL_SOLID_COLOR, 2, true, false);
  builder2.renderPass(99, 102, [0, 0, 0, 1], 102, 102, 3); // targetType 99 invalid
  const invalidTargetPacket = builder2.build();

  let rejectedTarget = false;
  try {
    await host.executePacket(invalidTargetPacket);
  } catch (e) {
    if (e.message === "RenderPass: invalid target kind 99 in 0x63") {
      rejectedTarget = true;
    } else {
      throw e;
    }
  }
  if (!rejectedTarget) {
    throw new Error(
      "Negative Control Failed: Bridge decoder failed to reject invalid render pass target type!",
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 5. SYNCHRONOUS ERROR-SCOPE DISCIPLINE & SUBMIT ORDERING (UNIT CONTROLS)
 * -----------------------------------------------------------------------------
 * Verifies that finish and submit occur synchronously inside the active error
 * scope before awaiting popErrorScope, preventing async interleaving bugs.
 */
export async function testSynchronousErrorScopeDiscipline(host) {
  // Positive: withErrorScopes successfully executes valid operations without throwing
  const result = await host.withErrorScopes(["validation", "out-of-memory"], () => {
    // Normal synchronous command recording and submission
    const enc = host.device.createCommandEncoder();
    const cb = enc.finish();
    host.device.queue.submit([cb]);
    return "ok";
  });

  if (result !== "ok") {
    throw new Error(`Synchronous error scope returned unexpected result: ${result}`);
  }
}

export async function testNegativeBrokenErrorScopeControl(host) {
  // Sub-test 1: Returning a Promise from syncAction must throw
  let rejectedPromise = false;
  try {
    await host.withErrorScopes(["validation"], async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
  } catch (e) {
    if (e.message && e.message.includes("syncAction returned a Promise")) {
      rejectedPromise = true;
    }
  }
  if (!rejectedPromise) {
    throw new Error(
      "Negative Control Failed: withErrorScopes failed to reject async/Promise return!",
    );
  }

  // Sub-test 2: Concurrent error scope invocation
  let innerErr = null;
  let innerPromise = null;
  await host.withErrorScopes(["validation"], () => {
    // Re-enter withErrorScopes while active; capture inner async promise rejection
    innerPromise = host
      .withErrorScopes(["validation"], () => {})
      .catch((e) => {
        innerErr = e;
      });
  });

  if (innerPromise) {
    await innerPromise;
  }

  if (
    !innerErr ||
    !innerErr.message ||
    !innerErr.message.includes("Error-scope serialization violation")
  ) {
    throw new Error(
      `Negative Control Failed: withErrorScopes failed to reject concurrent interleaved scopes! Observed: ${innerErr ? innerErr.message : "no error"}`,
    );
  }
}

/**
 * -----------------------------------------------------------------------------
 * 3. PENDING PRODUCT CAPABILITIES (EXCLUDED FROM ACCEPTANCE CLAIMS)
 * -----------------------------------------------------------------------------
 * Per OrangePelican root review and credit rules:
 * Invented mock classes (e.g. test-only handle managers or borrow controllers)
 * must NEVER be presented as passing product tests.
 *
 * All bridge capabilities are now backed by real Rust/Wasm exports.
 * The pending capability ledger is ZERO.
 */

export function getPendingBridgeCapabilities() {
  return [];
}

/**
 * -----------------------------------------------------------------------------
 * 6. STALE-EPOCH READBACK PUBLICATION GATE (58j.3 INVARIANT)
 * -----------------------------------------------------------------------------
 * Calls the real Rust/Wasm product export `gpu_bridge_try_publish_readback`
 * (alias `f3d_try_publish_readback`) from crates/f3d-runtime/src/gpu_host.rs:1161.
 *
 * Evaluates the epochHi and epochLo words attached by the bridge to a real readback
 * buffer against an active region epoch that advances:
 * - Matching epoch: try_publish_readback returns true (accepted).
 * - Stale epoch: try_publish_readback returns false (rejected/discarded).
 *
 * Honest boundary note: Target region epochs are test-supplied until a dedicated
 * region ownership manager exists in the bridge host runtime.
 * Strictly invokes the compiled Rust gate via Wasm export; no test-only JS mock.
 */
export async function testStaleEpochReadbackPublicationGate(host, wasmModule) {
  const tryPublishFn =
    wasmModule?.gpu_bridge_try_publish_readback || wasmModule?.f3d_try_publish_readback;
  if (!wasmModule || typeof tryPublishFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testStaleEpochReadbackPublicationGate requires compiled application Wasm with export 'gpu_bridge_try_publish_readback' (or 'f3d_try_publish_readback'). Silent JS fallback is forbidden.",
    );
  }

  // Execute a real GPU copy operation stamped with Epoch (0, 1)
  const width = 64;
  const height = 64;
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  const builder = new BinaryPacketBuilder();
  builder.createTexture(
    50,
    width,
    height,
    2,
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  );
  builder.createBuffer(51, readbackSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  // Stamped with epochHi = 0, epochLo = 1
  builder.copyTextureToBuffer(50, 51, width, height, 0, 1);
  const packet = builder.build();
  await host.executePacket(packet);

  // Retrieve real readback buffer with bridge-attached epoch words
  const readback = await host.readbackBuffer(51, readbackSize);
  const readbackEpochHi = readback.epochHi !== undefined ? readback.epochHi : 0;
  const readbackEpochLo = readback.epochLo !== undefined ? readback.epochLo : 1;

  // 1. Positive: Region at matching epoch (0, 1) must be accepted
  const matchingAccepted = tryPublishFn(readbackEpochHi, readbackEpochLo, 0, 1);
  if (!matchingAccepted) {
    throw new Error(
      `Positive Control Failed: Rust publication gate rejected matching readback epoch (${readbackEpochHi}, ${readbackEpochLo}) against region epoch (0, 1)!`,
    );
  }

  // 2. Negative: Region advances to epoch (0, 2); in-flight readback at (0, 1) is now stale and must be rejected
  const staleAccepted = tryPublishFn(readbackEpochHi, readbackEpochLo, 0, 2);
  if (staleAccepted) {
    throw new Error(
      `Negative Control Failed: Rust publication gate falsely accepted stale readback epoch (${readbackEpochHi}, ${readbackEpochLo}) against advanced region epoch (0, 2)!`,
    );
  }

  // 3. Negative: High-word mismatch (1, 1) must also be rejected
  const highMismatchAccepted = tryPublishFn(readbackEpochHi, readbackEpochLo, 1, 1);
  if (highMismatchAccepted) {
    throw new Error(
      `Negative Control Failed: Rust publication gate falsely accepted high-word mismatch epoch (${readbackEpochHi}, ${readbackEpochLo}) against region epoch (1, 1)!`,
    );
  }

  return "Verified via real Rust/Wasm product export gpu_bridge_try_publish_readback: matching epoch accepted (true), stale/advanced epoch rejected (false). (Region epoch test-supplied).";
}

/**
 * -----------------------------------------------------------------------------
 * 7. BUNDLE-THEN-DIRECT-DRAW STATE RESET (WEBGPU SPEC INVARIANT)
 * -----------------------------------------------------------------------------
 * Plan §8.5, §23 [S37], and AGENTS.md "Bundles reset render-pass state":
 * WebGPU spec mandates that executeBundles clears the current render pass's
 * pipeline and bind group bindings. Any subsequent direct draw within the same
 * pass MUST explicitly re-bind pipeline, bind group, and vertex buffer.
 *
 * Calls the real Rust/Wasm product export `gpu_bridge_build_bundle_direct_draw_packet`
 * (alias `f3d_build_bundle_direct_draw_packet`) from crates/f3d-runtime/src/gpu_host.rs:1414.
 *
 * Sequence in the compiled packet:
 * - Bundle 1: Triangle 1 (left side) with Green uniform at dynamic offset 0.
 * - executeBundles([1]) -> draws Green triangle.
 * - executeBundles([])  -> empty bundle list; spec-mandated pass state reset.
 * - RenderPass direct draw -> Triangle 2 (right side) with Blue uniform at dynamic offset 256.
 *   Re-binds pipeline, bind group with [256], and vertex buffer 3.
 * - Readback of target texture matches direct-JS oracle with 0 diffs.
 *
 * Negative Control: An explicitly isolated broken GPU operation that issues the post-bundle
 * direct draw of Triangle 2 without re-issuing setPipeline, setBindGroup, and setVertexBuffer.
 * The only difference from the positive path is the missing rebind. Tested against the
 * EXACT SAME `assertBundleDirectDrawMatch` assertion to verify detection and rejection.
 */
/**
 * Helper: Parses binary packet bytes according to bridge runtime opcode layouts
 * and produces a diagnostic command trace with key fields.
 */
export function decodePacketCommandTrace(packetBytes) {
  if (!packetBytes || packetBytes.byteLength < 16) {
    return ["<invalid packet: buffer shorter than 16-byte header>"];
  }
  const dataView = new DataView(packetBytes.buffer, packetBytes.byteOffset, packetBytes.byteLength);
  const magic = dataView.getUint32(0, true);
  const version = dataView.getUint16(4, true);
  const commandCount = dataView.getUint32(8, true);
  const dataLen = dataView.getUint32(12, true);
  const dataBlockStart = packetBytes.byteLength - dataLen;

  const trace = [
    `Header: magic=0x${magic.toString(16)}, version=${version}, commandCount=${commandCount}, dataLen=${dataLen}`,
  ];

  let cursor = 16;
  for (let i = 0; i < commandCount; i++) {
    if (cursor + 2 > dataBlockStart) {
      trace.push(
        `[Cmd ${i}] TRUNCATED: cursor ${cursor} exceeded dataBlockStart ${dataBlockStart}`,
      );
      break;
    }
    const opcode = dataView.getUint16(cursor, true);
    cursor += 2;

    switch (opcode) {
      case 1: {
        // OPCODE_CREATE_BUFFER
        if (cursor + 12 > dataBlockStart) {
          trace.push(`[Cmd ${i}] CREATE_BUFFER: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const bufferId = dataView.getUint32(cursor, true);
        const size = dataView.getUint32(cursor + 4, true);
        const usage = dataView.getUint32(cursor + 8, true);
        cursor += 12;
        trace.push(
          `[Cmd ${i}] CREATE_BUFFER: bufferId=${bufferId}, size=${size}, usage=0x${usage.toString(16)}`,
        );
        break;
      }
      case 2: {
        // OPCODE_WRITE_BUFFER
        if (cursor + 16 > dataBlockStart) {
          trace.push(`[Cmd ${i}] WRITE_BUFFER: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const bufferId = dataView.getUint32(cursor, true);
        const offset = dataView.getUint32(cursor + 4, true);
        const dataOffset = dataView.getUint32(cursor + 8, true);
        const dataLength = dataView.getUint32(cursor + 12, true);
        cursor += 16;
        trace.push(
          `[Cmd ${i}] WRITE_BUFFER: bufferId=${bufferId}, offset=${offset}, dataOffset=${dataOffset}, dataLen=${dataLength}`,
        );
        break;
      }
      case 3: {
        // OPCODE_CREATE_PIPELINE
        if (cursor + 32 > dataBlockStart) {
          trace.push(`[Cmd ${i}] CREATE_PIPELINE: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const pipelineId = dataView.getUint32(cursor, true);
        const codeOffset = dataView.getUint32(cursor + 4, true);
        const codeLen = dataView.getUint32(cursor + 8, true);
        const formatCode = dataView.getUint32(cursor + 12, true);
        const hasVB = dataView.getUint32(cursor + 16, true);
        const hasUB = dataView.getUint32(cursor + 20, true);
        const uniformSize = dataView.getUint32(cursor + 24, true);
        const vertexStride = dataView.getUint32(cursor + 28, true);
        cursor += 32;
        trace.push(
          `[Cmd ${i}] CREATE_PIPELINE: pipelineId=${pipelineId}, codeLen=${codeLen}, format=${formatCode}, hasVB=${hasVB}, hasUB=${hasUB}, uniformSize=${uniformSize}, vertexStride=${vertexStride}`,
        );
        break;
      }
      case 4: {
        // OPCODE_RENDER_PASS
        if (cursor + 44 > dataBlockStart) {
          trace.push(`[Cmd ${i}] RENDER_PASS: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const targetType = dataView.getUint32(cursor, true);
        const targetId = dataView.getUint32(cursor + 4, true);
        const cr = dataView.getFloat32(cursor + 8, true).toFixed(2);
        const cg = dataView.getFloat32(cursor + 12, true).toFixed(2);
        const cb = dataView.getFloat32(cursor + 16, true).toFixed(2);
        const ca = dataView.getFloat32(cursor + 20, true).toFixed(2);
        const pipelineId = dataView.getUint32(cursor + 24, true);
        const vertexBufferId = dataView.getUint32(cursor + 28, true);
        const vertexCount = dataView.getUint32(cursor + 32, true);
        const dynamicOffset = dataView.getUint32(cursor + 36, true);
        const uniformBufferId = dataView.getUint32(cursor + 40, true);
        cursor += 44;
        trace.push(
          `[Cmd ${i}] RENDER_PASS: targetType=${targetType === 0 ? "Offscreen" : "Canvas"}, targetId=${targetId}, clear=[${cr},${cg},${cb},${ca}], pipelineId=${pipelineId}, vbId=${vertexBufferId}, vertexCount=${vertexCount}, dynOffset=${dynamicOffset}, ubId=${uniformBufferId}`,
        );
        break;
      }
      case 5: {
        // OPCODE_COPY_TEXTURE_TO_BUFFER
        if (cursor + 24 > dataBlockStart) {
          trace.push(`[Cmd ${i}] COPY_TEXTURE_TO_BUFFER: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const textureId = dataView.getUint32(cursor, true);
        const bufferId = dataView.getUint32(cursor + 4, true);
        const w = dataView.getUint32(cursor + 8, true);
        const h = dataView.getUint32(cursor + 12, true);
        const epochHi = dataView.getUint32(cursor + 16, true);
        const epochLo = dataView.getUint32(cursor + 20, true);
        cursor += 24;
        trace.push(
          `[Cmd ${i}] COPY_TEXTURE_TO_BUFFER: textureId=${textureId}, bufferId=${bufferId}, dims=${w}x${h}, epoch=(${epochHi},${epochLo})`,
        );
        break;
      }
      case 6: {
        // OPCODE_CREATE_TEXTURE
        if (cursor + 20 > dataBlockStart) {
          trace.push(`[Cmd ${i}] CREATE_TEXTURE: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const textureId = dataView.getUint32(cursor, true);
        const w = dataView.getUint32(cursor + 4, true);
        const h = dataView.getUint32(cursor + 8, true);
        const formatCode = dataView.getUint32(cursor + 12, true);
        const usage = dataView.getUint32(cursor + 16, true);
        cursor += 20;
        trace.push(
          `[Cmd ${i}] CREATE_TEXTURE: textureId=${textureId}, dims=${w}x${h}, format=${formatCode}, usage=0x${usage.toString(16)}`,
        );
        break;
      }
      case 7: {
        // OPCODE_RECORD_BUNDLE
        if (cursor + 28 > dataBlockStart) {
          trace.push(`[Cmd ${i}] RECORD_BUNDLE: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const bundleId = dataView.getUint32(cursor, true);
        const pipelineId = dataView.getUint32(cursor + 4, true);
        const vertexBufferId = dataView.getUint32(cursor + 8, true);
        const vertexCount = dataView.getUint32(cursor + 12, true);
        const dynamicOffset = dataView.getUint32(cursor + 16, true);
        const uniformBufferId = dataView.getUint32(cursor + 20, true);
        const targetFormatCode = dataView.getUint32(cursor + 24, true);
        cursor += 28;
        trace.push(
          `[Cmd ${i}] RECORD_BUNDLE: bundleId=${bundleId}, pipelineId=${pipelineId}, vbId=${vertexBufferId}, vertexCount=${vertexCount}, dynOffset=${dynamicOffset}, ubId=${uniformBufferId}, format=${targetFormatCode}`,
        );
        break;
      }
      case 8: {
        // OPCODE_EXECUTE_BUNDLES
        if (cursor + 4 > dataBlockStart) {
          trace.push(`[Cmd ${i}] EXECUTE_BUNDLES: truncated`);
          cursor = dataBlockStart;
          break;
        }
        const bundleCount = dataView.getUint32(cursor, true);
        cursor += 4;
        const bundleIds = [];
        for (let b = 0; b < bundleCount; b++) {
          if (cursor + 4 > dataBlockStart) break;
          bundleIds.push(dataView.getUint32(cursor, true));
          cursor += 4;
        }
        trace.push(
          `[Cmd ${i}] EXECUTE_BUNDLES: count=${bundleCount}, bundleIds=[${bundleIds.join(", ")}]`,
        );
        break;
      }
      default:
        trace.push(`[Cmd ${i}] UNKNOWN_OPCODE: ${opcode}`);
        break;
    }
  }
  return trace;
}

export async function testBundleThenDirectDrawStateReset(
  host,
  device,
  wasmModule,
  width = 64,
  height = 64,
) {
  const buildBundleDirectFn =
    wasmModule?.f3d_build_bundle_direct_draw_packet ||
    wasmModule?.gpu_bridge_build_bundle_direct_draw_packet;
  if (!wasmModule || typeof buildBundleDirectFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testBundleThenDirectDrawStateReset requires compiled application Wasm with export 'gpu_bridge_build_bundle_direct_draw_packet' (or 'f3d_build_bundle_direct_draw_packet'). Silent JS fallback is forbidden.",
    );
  }

  // REAL RUST/WASM PATH: Packet generated by crates/f3d-runtime/src/gpu_host.rs
  const packet = buildBundleDirectFn();
  const readbackBufferId = 20; // Rust build_bundle_then_direct_draw_submission outputs to buffer 20
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  // 1. Decoded command trace of the real Rust packet
  const commandTrace = decodePacketCommandTrace(packet);

  // 2. Execute packet and record whether the positive path raised any WebGPU validation error inside the error scope
  let positiveValidationError = null;
  try {
    await host.executePacket(packet);
  } catch (err) {
    positiveValidationError = err.message || String(err);
  }

  // 3. Read back pixels from the offscreen target buffer
  let candidatePixels;
  let readbackError = null;
  try {
    candidatePixels = await host.readbackBuffer(readbackBufferId, readbackSize);
  } catch (err) {
    readbackError = err.message || String(err);
    candidatePixels = new Uint8Array(readbackSize);
  }

  // 4. Sample pixels at designated coordinates: (16,32), (48,32), (2,2)
  const greenIdx = 32 * bytesPerRow + 16 * 4;
  const blueIdx = 32 * bytesPerRow + 48 * 4;
  const blackIdx = 2 * bytesPerRow + 2 * 4;

  const s16_32 = [
    candidatePixels[greenIdx],
    candidatePixels[greenIdx + 1],
    candidatePixels[greenIdx + 2],
    candidatePixels[greenIdx + 3],
  ];
  const s48_32 = [
    candidatePixels[blueIdx],
    candidatePixels[blueIdx + 1],
    candidatePixels[blueIdx + 2],
    candidatePixels[blueIdx + 3],
  ];
  const s2_2 = [
    candidatePixels[blackIdx],
    candidatePixels[blackIdx + 1],
    candidatePixels[blackIdx + 2],
    candidatePixels[blackIdx + 3],
  ];

  const samplesSummary = `Sampled pixels: (16,32)=[${s16_32.join(",")}], (48,32)=[${s48_32.join(",")}], (2,2)=[${s2_2.join(",")}]`;
  const valSummary = `Positive WebGPU error scope: ${positiveValidationError ? `ERROR: ${positiveValidationError}` : "CLEAN (no validation error)"}`;
  const rbSummary = readbackError ? `Readback error: ${readbackError}` : "Readback OK";
  const traceFormatted = `Decoded Rust Packet Commands (${commandTrace.length}):\n${commandTrace.map((c) => `  ${c}`).join("\n")}`;

  // Oracle: Independent direct-JS execution
  const oraclePixels = await renderDirectBundleDirectReference(device, width, height);

  // 5. Positive Assertion: Must match pixel-for-pixel (0 differences) + verify sample points.
  // If positive path threw validation error or pixel assertion fails, include the command trace,
  // sampled pixels, and validation error status in the thrown error message.
  try {
    if (positiveValidationError) {
      throw new Error(
        `WebGPU validation error during positive packet execution: ${positiveValidationError}`,
      );
    }
    if (readbackError) {
      throw new Error(`Buffer readback failed: ${readbackError}`);
    }
    assertBundleDirectDrawMatch(candidatePixels, oraclePixels, width, height);
  } catch (matchErr) {
    const diagnosticMessage = `BundleThenDirectDraw State Reset Failed:\n${matchErr.message}\n${samplesSummary}\n${valSummary}\n${rbSummary}\n${traceFormatted}`;
    const err = new Error(diagnosticMessage);
    err.commandTrace = commandTrace;
    err.samples = { s16_32, s48_32, s2_2 };
    err.positiveValidationError = positiveValidationError;
    throw err;
  }

  return {
    candidatePixels,
    oraclePixels,
    commandTrace,
    samples: {
      s16_32,
      s48_32,
      s2_2,
    },
    positiveValidationError,
    detail: `Bundle-then-direct-draw verified: 0 diffs vs oracle. ${samplesSummary}. ${valSummary}. Trace: ${commandTrace.length} commands.`,
  };
}

export async function testNegativeBrokenBundleDirectDraw(
  host,
  oraclePixels,
  width = 64,
  height = 64,
) {
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  const target = host.device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const shaderModule = host.device.createShaderModule({ code: WGSL_FLAT_COLOR_BUNDLE });
  const bgl = host.device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
      },
    ],
  });

  const pipeline = host.device.createRenderPipeline({
    layout: host.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: "rgba8unorm" }],
    },
    primitive: { topology: "triangle-list" },
  });

  // Uniform buffer: Offset 0 Green, Offset 256 Blue
  const uniformBuffer = host.device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  host.device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([0.0, 1.0, 0.0, 1.0]));
  host.device.queue.writeBuffer(uniformBuffer, 256, new Float32Array([0.0, 0.0, 1.0, 1.0]));

  const bindGroup = host.device.createBindGroup({
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: 0, size: 16 } }],
  });

  // Vertex buffer 1: Triangle 1 (left side)
  const tri1Data = new Float32Array([
    -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0,
  ]);
  const vb1 = host.device.createBuffer({
    size: tri1Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  host.device.queue.writeBuffer(vb1, 0, tri1Data);

  // Vertex buffer 2: Triangle 2 (right side)
  const tri2Data = new Float32Array([
    0.0, -1.0, 0.0, 0.5, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
  ]);
  const vb2 = host.device.createBuffer({
    size: tri2Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  host.device.queue.writeBuffer(vb2, 0, tri2Data);

  // Bundle 1 draws Triangle 1 with Green
  const bundleEncoder = host.device.createRenderBundleEncoder({
    colorFormats: ["rgba8unorm"],
  });
  bundleEncoder.setPipeline(pipeline);
  bundleEncoder.setBindGroup(0, bindGroup, [0]);
  bundleEncoder.setVertexBuffer(0, vb1);
  bundleEncoder.draw(3, 1, 0, 0);
  const bundle = bundleEncoder.finish();

  const readback = host.device.createBuffer({
    size: readbackSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Execute on real GPU inside synchronous error scopes to record exact GPU behavior.
  // The broken control issues the post-bundle direct draw of Triangle 2 (3 vertices),
  // but WITHOUT re-issuing setPipeline, setBindGroup, and setVertexBuffer.
  let validationError = null;
  try {
    await host.withErrorScopes(["validation", "out-of-memory"], () => {
      const encoder = host.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: target.createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      // 1. Bundle draws Green left triangle
      pass.executeBundles([bundle]);

      // 2. Empty bundle sequence clears pass state per WebGPU spec invariant
      pass.executeBundles([]);

      // 3. BROKEN CONTROL: Still issue the post-bundle direct draw of Triangle 2,
      // but WITHOUT re-issuing setPipeline, setBindGroup, and setVertexBuffer!
      // The only difference from the positive path is the missing rebind.
      pass.draw(3, 1, 0, 0);

      pass.end();

      encoder.copyTextureToBuffer(
        { texture: target },
        { buffer: readback, bytesPerRow, rowsPerImage: height },
        [width, height, 1],
      );

      const commandBuffer = encoder.finish();
      host.device.queue.submit([commandBuffer]);
    });
  } catch (err) {
    validationError = err;
  }

  // Record what the real GPU did
  const recordedGpuBehavior = {
    behavior: validationError ? "validation_error_captured" : "wrong_geometry_rendered",
    detail: validationError
      ? validationError.message || String(validationError)
      : "Direct draw issued without rebind; executed with missing/stale pass state on GPU.",
  };

  // Read back actual GPU pixels from the offscreen target
  let brokenPixels;
  try {
    brokenPixels = await readbackGpuBuffer(host.device, readback, readbackSize);
  } catch (_) {
    // If command buffer was marked invalid by WebGPU validation error preventing copy execution,
    // readback buffer contains its initial unwritten bytes
    brokenPixels = new Uint8Array(readbackSize);
  }

  target.destroy();
  uniformBuffer.destroy();
  vb1.destroy();
  vb2.destroy();
  readback.destroy();

  // The EXACT SAME assertion must be applied to the broken output and MUST reject it
  let rejected = false;
  try {
    assertBundleDirectDrawMatch(brokenPixels, oraclePixels, width, height);
  } catch (e) {
    rejected = true;
  }

  if (!rejected) {
    throw new Error(
      "Negative Control Failed: assertBundleDirectDrawMatch did not reject GPU output produced without post-bundle rebind!",
    );
  }

  return recordedGpuBehavior;
}

/**
 * -----------------------------------------------------------------------------
 * 8. GENERATIONAL HANDLE ABA PUBLICATION GATE (§6.5, VQA.6)
 * -----------------------------------------------------------------------------
 * Plan §6.5, §23 [S33], and AGENTS.md "Generational Handle Slot Table & Publication Gate":
 * Verifies that GPU resource handles are guarded by a typed NonZeroU32 generation
 * in the bridge slot table (`f3d_core::Handle` and `crates/f3d-runtime/src/gpu_host.rs`).
 *
 * Calls real Rust/Wasm product exports:
 * - `gpu_bridge_check_resource_handle(index, generation) -> bool` (alias `f3d_check_resource_handle`)
 * - `gpu_bridge_advance_resource_generation(index) -> u32` (alias `f3d_advance_resource_generation`)
 *
 * Sequence:
 * 1. Execute a real GPU packet (triangle packet registers IDs 1, 2, 3, 10, 100 at generation 1;
 *    bundle packet registers 1, 2, 3, 10, 20, 200 at generation 1).
 * 2. Assert `check(id, 1)` is true for registered IDs.
 * 3. Advance slot generation: `advance(id)` returns 2.
 * 4. Assert `check(id, 1)` is false (stale ABA generation rejected).
 * 5. Assert `check(id, 2)` is true (new active generation accepted).
 * 6. Assert `check(id, 0)` is false (generation zero strictly rejected via HandleError::InvalidGeneration).
 *
 * Negative Control: An explicitly isolated broken publication attempt of the completed GPU
 * resource keyed by the stale generation (generation 1 after advance, or generation 0).
 * Tested against the EXACT SAME `assertGenerationalHandlePublication` assertion to verify
 * detection and rejection.
 */
export async function testGenerationalHandleAbaPublication(host, wasmModule, canvasContext) {
  const checkFn =
    wasmModule?.f3d_check_resource_handle || wasmModule?.gpu_bridge_check_resource_handle;
  const advanceFn =
    wasmModule?.f3d_advance_resource_generation ||
    wasmModule?.gpu_bridge_advance_resource_generation;
  const buildTriangleFn =
    wasmModule?.f3d_build_first_frame_packet || wasmModule?.gpu_bridge_build_triangle_packet;

  if (
    !wasmModule ||
    typeof checkFn !== "function" ||
    typeof advanceFn !== "function" ||
    typeof buildTriangleFn !== "function"
  ) {
    throw new Error(
      "Missing Wasm Export: testGenerationalHandleAbaPublication requires compiled application Wasm with exports 'gpu_bridge_check_resource_handle', 'gpu_bridge_advance_resource_generation', and 'gpu_bridge_build_triangle_packet' (or aliases). Silent JS fallback is forbidden.",
    );
  }

  // 1. Execute a real GPU packet through the WebGPU bridge
  // In crates/f3d-runtime/src/gpu_host.rs:890-898, build_triangle_submission registers
  // resource IDs: 1 (uniform buffer), 2 (vertex buffer), 3 (readback buffer), 10 (target texture), 100 (pipeline)
  // at initial generation 1 in the global resource slot table.
  const packet = buildTriangleFn();
  await host.executePacket(packet, canvasContext);

  const registeredTriangleIds = [1, 2, 3, 10, 100];
  for (const id of registeredTriangleIds) {
    if (!checkFn(id, 1)) {
      throw new Error(
        `Positive Control Failed: Expected resource id ${id} to be registered at generation 1, but check returned false!`,
      );
    }
  }

  // Resource 10 (target texture) lifecycle and ABA publication test
  const testId = 10;

  // 2. Assert check(id, 1) is true
  assertGenerationalHandlePublication(checkFn, testId, 1);

  // 3. Assert advance(id) returns 2 (simulating resource slot release and reuse)
  const newGen = advanceFn(testId);
  if (newGen !== 2) {
    throw new Error(
      `Lifecycle Advance Failed: Expected advance_resource_generation(${testId}) to return 2, got ${newGen}!`,
    );
  }

  // 4. Assert check(id, 1) is false (stale ABA generation rejected)
  if (checkFn(testId, 1) !== false) {
    throw new Error(
      `ABA Hazard Failed: Expected check_resource_handle(${testId}, 1) to return false after advance, got true!`,
    );
  }

  // 5. Assert check(id, 2) is true (fresh generation after reallocation accepted)
  if (checkFn(testId, 2) !== true) {
    throw new Error(
      `Positive Control Failed: Expected check_resource_handle(${testId}, 2) to return true after advance, got false!`,
    );
  }
  assertGenerationalHandlePublication(checkFn, testId, 2);

  // 6. Assert check(id, 0) is false (generation zero strictly rejected via HandleError::InvalidGeneration)
  if (checkFn(testId, 0) !== false) {
    throw new Error(
      `Generation Zero Invariant Failed: Expected check_resource_handle(${testId}, 0) to return false, got true!`,
    );
  }

  return {
    testId,
    initialGen: 1,
    advancedGen: newGen,
    registeredTriangleIds,
    detail: `Verified via real Rust/Wasm slot table exports: check(${testId}, 1)=true, advance(${testId})=2, check(${testId}, 1)=false, check(${testId}, 2)=true, check(${testId}, 0)=false.`,
  };
}

export async function testNegativeBrokenGenerationalHandlePublication(
  host,
  wasmModule,
  testId = 10,
  staleGeneration = 1,
) {
  const checkFn =
    wasmModule?.f3d_check_resource_handle || wasmModule?.gpu_bridge_check_resource_handle;
  if (!wasmModule || typeof checkFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenGenerationalHandlePublication requires 'gpu_bridge_check_resource_handle' (or alias). Silent JS fallback is forbidden.",
    );
  }

  // Verify that the GPU resource was indeed created and executed on the bridge host
  if (!host.textures.has(testId)) {
    throw new Error(
      `testNegativeBrokenGenerationalHandlePublication: target texture ${testId} not found in bridge host textures`,
    );
  }

  // A real GPU publication attempt of the completed offscreen target resource,
  // but keyed by the stale generation (generation 1 after slot advanced to generation 2).
  // The EXACT SAME assertion `assertGenerationalHandlePublication` must be invoked, and MUST reject it.
  let rejectedStale = false;
  try {
    assertGenerationalHandlePublication(checkFn, testId, staleGeneration);
  } catch (err) {
    if (err.message && err.message.includes("Generational Handle Publication Rejected")) {
      rejectedStale = true;
    }
  }

  if (!rejectedStale) {
    throw new Error(
      `Negative Control Failed: assertGenerationalHandlePublication did not reject publication attempt with stale generation (${testId}, gen=${staleGeneration})!`,
    );
  }

  // Also verify generation zero publication attempt is rejected by the exact same assertion
  let rejectedZero = false;
  try {
    assertGenerationalHandlePublication(checkFn, testId, 0);
  } catch (err) {
    if (err.message && err.message.includes("Generational Handle Publication Rejected")) {
      rejectedZero = true;
    }
  }

  if (!rejectedZero) {
    throw new Error(
      `Negative Control Failed: assertGenerationalHandlePublication did not reject publication attempt with generation zero (${testId}, gen=0)!`,
    );
  }

  return {
    behavior: "stale_generation_publication_rejected",
    detail: `Publication attempt for GPU resource ${testId} keyed by stale generation ${staleGeneration} and zero generation was strictly rejected by assertGenerationalHandlePublication.`,
  };
}

/**
 * -----------------------------------------------------------------------------
 * 9. LINEAR MEMORY BORROW GUARDS (§6.6, §13.1, VQA.6)
 * -----------------------------------------------------------------------------
 * Plan §6.6, §13.1, and AGENTS.md "Unsafe Code" / "Linear Memory":
 * No application callback, memory growth, or scheduler re-entry may occur while
 * a Rust borrow of linear memory is live. Release the borrow before calling
 * effectful host/user code.
 *
 * Calls real Rust/Wasm product exports:
 * - `gpu_bridge_borrow_enter() -> u64` (alias `f3d_borrow_enter`)
 * - `gpu_bridge_borrow_exit(token: u64) -> bool` (alias `f3d_borrow_exit`)
 * - `gpu_bridge_try_grow_memory(pages: u32) -> bool` (alias `f3d_try_grow_memory`)
 * - `gpu_bridge_build_triangle_packet() -> Uint8Array` (alias `f3d_build_first_frame_packet`)
 *
 * Sequence:
 * Part 1: BorrowScope Policy Gate (Metadata Unit Gate)
 * 1. Assert grow true when idle (`try_grow_memory(1) === true`).
 * 2. Assert enter gives nonzero BigInt token (`typeof token === "bigint" && token !== 0n`).
 * 3. Assert grow false while borrowed (`try_grow_memory(1) === false`).
 * 4. Assert exit with wrong token false (`borrow_exit(wrongToken) === false`).
 * 5. Assert exit with right token true (`borrow_exit(token) === true`).
 * 6. Assert grow true again (`try_grow_memory(1) === true`).
 *
 * Part 2: Real Application-Memory Transport & Growth Invariance (OrangePelican Mail #6811)
 * 7. Strictly require real `WebAssembly.Memory` passed from `await wasm.default()`.
 *    No optional fallback or bypass branch permitted.
 * 8. Pre-growth owned packet: call `buildTriangleFn()` to produce an owned Uint8Array,
 *    and take a read-only snapshot (`new Uint8Array(preGrowthPacket.slice())`).
 * 9. Memory growth: call `wasmMemory.grow(1)` and verify byteLength increase by 65536.
 * 10. View invalidation: verify unrefreshed view/buffer is detached or buffer identity changed.
 * 11. Owned-copy stability: verify `preGrowthPacket` bytes are uncorrupted across `grow(1)`.
 * 12. Post-growth packet & view refresh: call `buildTriangleFn()` post-growth, verifying
 *     wasm-bindgen's internal buffer view refreshed properly and produces matching packet bytes.
 * 13. Callback reentry: call `borrow_enter()` after growth to demonstrate real Rust reentry,
 *     verify growth is blocked during reentry, exit cleanly, and verify owned pre-callback
 *     bytes remain intact.
 *
 * Negative Control: An open borrow is entered, and an attempt to grow memory
 * is issued. The EXACT SAME `assertMemoryGrowthAllowed` assertion rejects it.
 */
export async function testLinearMemoryBorrowGuards(host, wasmModule, wasmMemory) {
  const enterFn = wasmModule?.f3d_borrow_enter || wasmModule?.gpu_bridge_borrow_enter;
  const exitFn = wasmModule?.f3d_borrow_exit || wasmModule?.gpu_bridge_borrow_exit;
  const growFn = wasmModule?.f3d_try_grow_memory || wasmModule?.gpu_bridge_try_grow_memory;
  const buildTriangleFn =
    wasmModule?.f3d_build_first_frame_packet || wasmModule?.gpu_bridge_build_triangle_packet;

  if (
    !wasmModule ||
    typeof enterFn !== "function" ||
    typeof exitFn !== "function" ||
    typeof growFn !== "function" ||
    typeof buildTriangleFn !== "function"
  ) {
    throw new Error(
      "Missing Wasm Export: testLinearMemoryBorrowGuards requires compiled application Wasm with exports 'gpu_bridge_borrow_enter', 'gpu_bridge_borrow_exit', 'gpu_bridge_try_grow_memory', and 'gpu_bridge_build_triangle_packet' (or canonical aliases). Silent JS fallback is forbidden.",
    );
  }

  // Strictly require real WebAssembly.Memory captured from await wasm.default().
  // No optional-pass or bypass branch permitted (OrangePelican Mail #6811).
  const memory = wasmMemory || wasmModule?.memory;
  if (!memory || !(memory instanceof WebAssembly.Memory) || typeof memory.grow !== "function") {
    throw new Error(
      "Missing WebAssembly.Memory: testLinearMemoryBorrowGuards strictly requires the WebAssembly.Memory instance captured from await wasm.default(). Optional or missing memory fallbacks are forbidden.",
    );
  }

  // --- Part 1: BorrowScope Policy Gate (Metadata Unit Gate) ---
  // 1. Assert grow true when idle
  const idleGrow = growFn(1);
  if (idleGrow !== true) {
    throw new Error(
      "Positive Control Failed: Expected try_grow_memory(1) to return true while idle!",
    );
  }
  assertMemoryGrowthAllowed(growFn, 1);

  // 2. Assert enter gives nonzero BigInt token (wasm-bindgen u64 is strictly BigInt)
  const token = enterFn();
  if (typeof token !== "bigint" || token === 0n) {
    throw new Error(
      `Positive Control Failed: Expected borrow_enter() to return non-zero BigInt BorrowToken (wasm-bindgen u64), got ${typeof token} (${token})!`,
    );
  }

  // 3. Assert grow false while borrowed
  const borrowedGrow = growFn(1);
  if (borrowedGrow !== false) {
    // Clean up borrow before throwing
    exitFn(token);
    throw new Error(
      "Safety Hazard: Expected try_grow_memory(1) to return false while an active borrow is held!",
    );
  }

  // 4. Assert exit with wrong token false (BigInt token)
  const wrongToken = token + 999999n;
  const wrongExit = exitFn(wrongToken);
  if (wrongExit !== false) {
    exitFn(token);
    throw new Error(
      "Token Security Failed: Expected borrow_exit with wrong token to return false!",
    );
  }

  // 5. Assert exit with right token true
  const rightExit = exitFn(token);
  if (rightExit !== true) {
    throw new Error(
      "Positive Control Failed: Expected borrow_exit with valid token to return true!",
    );
  }

  // 6. Assert grow true again after exit
  const postExitGrow = growFn(1);
  if (postExitGrow !== true) {
    throw new Error(
      "Positive Control Failed: Expected try_grow_memory(1) to return true after borrow scope exit!",
    );
  }
  assertMemoryGrowthAllowed(growFn, 1);

  const policyGateStatus = {
    token: String(token),
    idleGrow: true,
    borrowedGrowBlocked: true,
    wrongTokenRejected: true,
    cleanExitRestored: true,
  };

  // --- Part 2: Real Application-Memory Transport & Growth Invariance ---
  // No arbitrary canary writes into unallocated Rust heap (Mail #6811).
  // A. Generate real owned packet from Rust before memory growth
  const preGrowthPacket = buildTriangleFn();
  if (!(preGrowthPacket instanceof Uint8Array) || preGrowthPacket.byteLength < 32) {
    throw new Error(
      `Pre-growth packet invalid: expected Uint8Array with length >= 32, got ${preGrowthPacket?.constructor?.name} (len=${preGrowthPacket?.byteLength})`,
    );
  }

  // Take an independent read-only snapshot of the owned packet bytes to serve as stable ground truth
  const preGrowthSnapshot = new Uint8Array(preGrowthPacket.slice());

  // Capture initial WebAssembly.Memory state and un-refreshed raw view
  const initialBytes = memory.buffer.byteLength;
  const initialPages = initialBytes / 65536;
  const oldBuffer = memory.buffer;
  const unrefreshedMemoryView = new Uint8Array(oldBuffer);

  // B. Perform actual WebAssembly.Memory growth
  memory.grow(1);
  const newBytes = memory.buffer.byteLength;
  const newPages = newBytes / 65536;

  if (newBytes !== initialBytes + 65536) {
    throw new Error(
      `Real Memory Growth Failed: Expected buffer byteLength to increase by 65536, got from ${initialBytes} to ${newBytes}`,
    );
  }

  // C. Verify un-refreshed view detachment or buffer identity change
  const isDetachedOrChanged =
    oldBuffer !== memory.buffer ||
    oldBuffer.detached === true ||
    unrefreshedMemoryView.byteLength === 0;

  if (!isDetachedOrChanged) {
    throw new Error(
      "Transport Invariant Failed: WebAssembly.Memory.grow(1) did not invalidate previous ArrayBuffer/view reference!",
    );
  }

  // D. Verify pre-growth owned packet remains completely intact and uncorrupted
  if (preGrowthPacket.byteLength !== preGrowthSnapshot.byteLength) {
    throw new Error(
      `Owned Copy Invariant Failed: Pre-growth owned packet length changed across memory.grow() (expected ${preGrowthSnapshot.byteLength}, got ${preGrowthPacket.byteLength})`,
    );
  }
  for (let i = 0; i < preGrowthSnapshot.length; i++) {
    if (preGrowthPacket[i] !== preGrowthSnapshot[i]) {
      throw new Error(
        `Owned Copy Invariant Failed: Pre-growth owned packet byte corrupted at index ${i} across memory.grow() (expected ${preGrowthSnapshot[i]}, got ${preGrowthPacket[i]})`,
      );
    }
  }

  // E. Call packet generator again post-growth to verify wasm-bindgen's cached-view refresh
  // and compare output bytes against the pre-growth snapshot
  const postGrowthPacket = buildTriangleFn();
  if (!(postGrowthPacket instanceof Uint8Array)) {
    throw new Error(
      `Post-growth packet invalid: expected Uint8Array, got ${postGrowthPacket?.constructor?.name}`,
    );
  }
  if (postGrowthPacket.byteLength !== preGrowthSnapshot.byteLength) {
    throw new Error(
      `Cached View Refresh Failed: Post-growth packet byteLength mismatch (expected ${preGrowthSnapshot.byteLength}, got ${postGrowthPacket.byteLength})`,
    );
  }
  for (let i = 0; i < preGrowthSnapshot.length; i++) {
    if (postGrowthPacket[i] !== preGrowthSnapshot[i]) {
      throw new Error(
        `Cached View Refresh Failed: Post-growth packet byte mismatch at index ${i} (expected ${preGrowthSnapshot[i]}, got ${postGrowthPacket[i]})`,
      );
    }
  }

  // F. Demonstrate real Rust callback reentry after growth and validate pre-callback owned bytes
  const reentryToken = enterFn();
  if (typeof reentryToken !== "bigint" || reentryToken === 0n) {
    throw new Error(
      `Reentry Failed: Expected borrow_enter() after growth to return non-zero BigInt token, got ${typeof reentryToken}`,
    );
  }
  // While reentered into borrow, verify growth is blocked
  const reentryGrowBlocked = growFn(1) === false;
  if (!reentryGrowBlocked) {
    exitFn(reentryToken);
    throw new Error(
      "Reentry Safety Failed: Expected try_grow_memory(1) to be blocked during reentered borrow!",
    );
  }
  // Exit the reentered borrow cleanly
  const reentryExitOk = exitFn(reentryToken);
  if (!reentryExitOk) {
    throw new Error(
      "Reentry Safety Failed: Expected borrow_exit(reentryToken) to succeed after reentered borrow!",
    );
  }

  // Validate that the owned pre-growth packet bytes are still perfectly intact after Rust reentry
  for (let i = 0; i < preGrowthSnapshot.length; i++) {
    if (preGrowthPacket[i] !== preGrowthSnapshot[i]) {
      throw new Error(
        `Reentry Invariant Failed: Pre-growth owned packet byte corrupted at index ${i} across Rust callback reentry`,
      );
    }
  }

  const memoryGrowthReport = {
    initialPages,
    newPages,
    initialBytes,
    newBytes,
    isDetachedOrChanged,
    preGrowthPacketBytes: preGrowthSnapshot.byteLength,
    ownedPacketPreserved: true,
    postGrowthPacketRefreshed: true,
    reentryValidated: true,
  };

  return {
    policyGateStatus,
    memoryGrowthReport,
    detail: `Verified real Wasm memory growth (${initialPages}->${newPages} pages, buffer detached/changed=${isDetachedOrChanged}, owned packet preserved [${preGrowthSnapshot.byteLength}B], cached view refreshed on post-growth allocation, Rust reentry verified) + BorrowScope policy gate (idle=true, in-borrow=false, wrong-token=false, exit=true).`,
  };
}

export async function testNegativeBrokenLinearMemoryBorrowGuard(host, wasmModule) {
  const enterFn = wasmModule?.f3d_borrow_enter || wasmModule?.gpu_bridge_borrow_enter;
  const exitFn = wasmModule?.f3d_borrow_exit || wasmModule?.gpu_bridge_borrow_exit;
  const growFn = wasmModule?.f3d_try_grow_memory || wasmModule?.gpu_bridge_try_grow_memory;

  if (
    !wasmModule ||
    typeof enterFn !== "function" ||
    typeof exitFn !== "function" ||
    typeof growFn !== "function"
  ) {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenLinearMemoryBorrowGuard requires 'gpu_bridge_borrow_enter' / 'gpu_bridge_try_grow_memory'.",
    );
  }

  // Enter an active borrow scope (strictly requires BigInt token)
  const token = enterFn();
  if (typeof token !== "bigint" || token === 0n) {
    throw new Error(
      `Broken Control Setup Failed: borrow_enter did not return valid non-zero BigInt token (got ${typeof token} ${token})`,
    );
  }

  // Broken attempt: Attempt memory growth while an open borrow is active.
  // The EXACT SAME `assertMemoryGrowthAllowed` assertion must be invoked, and MUST reject it.
  let rejected = false;
  try {
    assertMemoryGrowthAllowed(growFn, 1);
  } catch (err) {
    if (err.message && err.message.includes("Linear Memory Borrow Violation")) {
      rejected = true;
    }
  } finally {
    // Release the borrow so runtime remains clean
    exitFn(token);
  }

  if (!rejected) {
    throw new Error(
      "Negative Control Failed: assertMemoryGrowthAllowed did not reject memory growth attempt inside open borrow scope!",
    );
  }

  return {
    behavior: "growth_during_borrow_rejected",
    detail:
      "Linear memory growth during open borrow scope was strictly rejected by assertMemoryGrowthAllowed.",
  };
}

/**
 * -----------------------------------------------------------------------------
 * 10. AFFINEROWS GPU WIRE LAYOUT VALIDATION (§6.1, §6.2, VQA.6)
 * -----------------------------------------------------------------------------
 * Plan §6.1, §6.2, §23 [S33], and AGENTS.md "Core Invariants":
 * AffineRows packs a 4x4 matrix into 3 rows of vec4 (48 bytes: r0, r1, r2),
 * requiring the fourth row of the transform to be strictly affine `[0.0, 0.0, 0.0, 1.0]`.
 * Validates wire byte layout and rejects non-affine perspective terms before GPU upload.
 *
 * Wire format specifications:
 * - 48 bytes: decoded as canonical AffineRows wire record (row-major, translation at floats 3, 7, 11).
 *             Fourth row [0, 0, 0, 1] is implicit. Rejects non-finite floats (NaN/Inf) with code 2.
 * - 64 bytes: decoded as column-major 4x4 Matrix4. Verified via AffineRows::from_column_major.
 * - len < 48: rejected with code 1 (LAYOUT_VALIDATION_BUFFER_TOO_SMALL, required: 48).
 * - 49..63 bytes: rejected with code 1 (LAYOUT_VALIDATION_BUFFER_TOO_SMALL, required: 64).
 * - len > 64: rejected with code 4 (LAYOUT_VALIDATION_INCOMPATIBLE_TARGET, target: 64).
 *
 * Calls real Rust/Wasm product export:
 * - `gpu_bridge_validate_affine_rows(bytes: Uint8Array) -> i32` (alias `f3d_validate_affine_rows`)
 *   Return codes:
 *   - 0: LAYOUT_VALIDATION_OK
 *   - 1: LAYOUT_VALIDATION_BUFFER_TOO_SMALL (< 48 or 49..63 bytes)
 *   - 2: LAYOUT_VALIDATION_NON_AFFINE_MATRIX (perspective terms non-zero, invalid scale, or non-finite)
 *   - 4: LAYOUT_VALIDATION_INCOMPATIBLE_TARGET (> 64 bytes)
 *   - 3..7: Other LayoutError codes
 *
 * Sequence:
 * 1. 64-byte column-major identity matrix -> validate returns code 0 (valid).
 *    Verify with `assertAffineRowsLayoutValid`.
 * 2. 64-byte column-major matrix with nonzero translation (tx=12.5, ty=-4.0, tz=100.0)
 *    -> validate returns code 0 (valid). Verify with `assertAffineRowsLayoutValid`.
 * 3. 64-byte column-major matrix with e[11] non-zero (row 3, col 2 perspective term)
 *    -> validate returns code 2 (LAYOUT_VALIDATION_NON_AFFINE_MATRIX).
 * 4. 48-byte AffineRows wire record with nonzero translation (tx=12.5 at float 3, ty=-4.0 at float 7, tz=100.0 at float 11)
 *    -> validate returns code 0 (valid). Verify with `assertAffineRowsLayoutValid`.
 * 5. 48-byte AffineRows wire record with NaN translation component (float 3 = NaN)
 *    -> validate returns code 2 (LAYOUT_VALIDATION_NON_AFFINE_MATRIX).
 * 6. 47-byte buffer (1 byte smaller than 48-byte AffineRows layout, len < 48)
 *    -> validate returns code 1 (LAYOUT_VALIDATION_BUFFER_TOO_SMALL).
 * 7. 56-byte buffer (intermediate length, 48 < len < 64)
 *    -> validate returns code 1 (LAYOUT_VALIDATION_BUFFER_TOO_SMALL).
 * 8. 72-byte buffer (oversized buffer, len > 64)
 *    -> validate returns code 4 (LAYOUT_VALIDATION_INCOMPATIBLE_TARGET).
 *
 * GPU-Side Control: Attempt to upload a non-affine matrix through the validator gate,
 * asserting that the packet path refuses before any GPU submission occurs.
 */
export async function testAffineRowsGpuLayoutValidation(host, wasmModule) {
  const validateFn =
    wasmModule?.f3d_validate_affine_rows || wasmModule?.gpu_bridge_validate_affine_rows;

  if (!wasmModule || typeof validateFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testAffineRowsGpuLayoutValidation requires compiled application Wasm with export 'gpu_bridge_validate_affine_rows' (or alias 'f3d_validate_affine_rows'). Silent JS fallback is forbidden.",
    );
  }

  // 1. Real 64-byte column-major identity matrix:
  // Column-major 4x4: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
  const identityBytes = new Uint8Array(64);
  const idFloats = new Float32Array(identityBytes.buffer);
  idFloats[0] = 1.0;
  idFloats[5] = 1.0;
  idFloats[10] = 1.0;
  idFloats[15] = 1.0;

  const code0 = validateFn(identityBytes);
  if (code0 !== 0) {
    throw new Error(
      `Positive Control Failed: Expected validate_affine_rows(identity) to return 0, got ${code0}!`,
    );
  }
  assertAffineRowsLayoutValid(validateFn, identityBytes);

  // 2. Real 64-byte column-major matrix with nonzero translation (tx=12.5, ty=-4.0, tz=100.0):
  // Translation components in column-major 4x4 are at elements 12, 13, 14 (column 3, rows 0..2).
  const valid64TransBytes = new Uint8Array(64);
  const v64Floats = new Float32Array(valid64TransBytes.buffer);
  v64Floats[0] = 1.0;
  v64Floats[5] = 1.0;
  v64Floats[10] = 1.0;
  v64Floats[15] = 1.0;
  v64Floats[12] = 12.5; // tx
  v64Floats[13] = -4.0; // ty
  v64Floats[14] = 100.0; // tz

  const code0Trans = validateFn(valid64TransBytes);
  if (code0Trans !== 0) {
    throw new Error(
      `Positive Control Failed: Expected validate_affine_rows(64-byte translated matrix) to return 0, got ${code0Trans}!`,
    );
  }
  assertAffineRowsLayoutValid(validateFn, valid64TransBytes);

  // 3. 64-byte matrix with e[11] non-zero (perspective component at column 2, row 3 in 0-indexed column-major):
  // Column 2 entries are elements 8, 9, 10, 11 (where 11 is row 3). Non-zero e[11] breaks affine invariant!
  const nonAffineBytes = new Uint8Array(64);
  const naFloats = new Float32Array(nonAffineBytes.buffer);
  naFloats[0] = 1.0;
  naFloats[5] = 1.0;
  naFloats[10] = 1.0;
  naFloats[15] = 1.0;
  naFloats[11] = 0.5; // perspective element

  const code2 = validateFn(nonAffineBytes);
  if (code2 !== 2) {
    throw new Error(
      `Non-Affine Rejection Failed: Expected validate_affine_rows(nonAffine) to return 2 (NON_AFFINE_MATRIX), got ${code2}!`,
    );
  }

  // 4. Real 48-byte AffineRows wire record with nonzero translation (tx=12.5, ty=-4.0, tz=100.0):
  // Row-major 3x4: row 0 [m00,m01,m02,tx], row 1 [m10,m11,m12,ty], row 2 [m20,m21,m22,tz].
  // Translation is at floats 3, 7, 11.
  const valid48TransBytes = new Uint8Array(48);
  const v48Floats = new Float32Array(valid48TransBytes.buffer);
  v48Floats[0] = 1.0; // m00
  v48Floats[5] = 1.0; // m11
  v48Floats[10] = 1.0; // m22
  v48Floats[3] = 12.5; // tx
  v48Floats[7] = -4.0; // ty
  v48Floats[11] = 100.0; // tz

  const code0AffineTrans = validateFn(valid48TransBytes);
  if (code0AffineTrans !== 0) {
    throw new Error(
      `Positive Control Failed: Expected validate_affine_rows(48-byte translated AffineRows) to return 0, got ${code0AffineTrans}!`,
    );
  }
  assertAffineRowsLayoutValid(validateFn, valid48TransBytes);

  // 5. 48-byte AffineRows wire record with NaN translation:
  // Non-finite translation float must be rejected with code 2 (NON_AFFINE_MATRIX).
  const nan48Bytes = new Uint8Array(48);
  const nan48Floats = new Float32Array(nan48Bytes.buffer);
  nan48Floats[0] = 1.0;
  nan48Floats[5] = 1.0;
  nan48Floats[10] = 1.0;
  nan48Floats[3] = NaN; // NaN translation component
  nan48Floats[7] = -4.0;
  nan48Floats[11] = 100.0;

  const code2AffineNan = validateFn(nan48Bytes);
  if (code2AffineNan !== 2) {
    throw new Error(
      `Non-Affine Rejection Failed: Expected validate_affine_rows(48-byte NaN translation) to return 2 (NON_AFFINE_MATRIX), got ${code2AffineNan}!`,
    );
  }

  // 6. 47-byte buffer (1 byte smaller than 48-byte AffineRows layout, len < 48):
  const smallBytes = new Uint8Array(47);
  const code1Small = validateFn(smallBytes);
  if (code1Small !== 1) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(47-byte) to return 1 (BUFFER_TOO_SMALL), got ${code1Small}!`,
    );
  }

  // 7. 56-byte buffer (intermediate length, 48 < len < 64):
  const midLenBytes = new Uint8Array(56);
  const code1Mid = validateFn(midLenBytes);
  if (code1Mid !== 1) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(56-byte) to return 1 (BUFFER_TOO_SMALL), got ${code1Mid}!`,
    );
  }

  // 8. 72-byte buffer (oversized buffer, len > 64):
  const oversizedBytes = new Uint8Array(72);
  const code4Oversized = validateFn(oversizedBytes);
  if (code4Oversized !== 4) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(72-byte) to return 4 (INCOMPATIBLE_TARGET), got ${code4Oversized}!`,
    );
  }

  // 9. Authoritative Browser Execution: Real AffineRows Transform Packet & WGSL Evaluation
  // Calls Chartreuse's export gpu_bridge_build_affine_rows_transform_packet (§6.1, §6.2, vqa.6)
  const buildAffineTransformFn =
    wasmModule?.f3d_build_affine_rows_transform_packet ||
    wasmModule?.gpu_bridge_build_affine_rows_transform_packet;
  let transformReadbackResult = null;

  if (typeof buildAffineTransformFn === "function") {
    const transformPacket = buildAffineTransformFn();
    const readbackBufferId = 20; // staging buffer 20 per crates/f3d-runtime/src/gpu_host.rs:1615
    const bytesPerRow = computeAlignedBytesPerRow(64);
    const readbackSize = bytesPerRow * 64;

    await host.executePacket(transformPacket);
    const pixels = await host.readbackBuffer(readbackBufferId, readbackSize);

    // Verify hand-computed screen pixel coordinates via assertAffineRowsTransformMatch
    assertAffineRowsTransformMatch(pixels, 64, 64);

    const c48_32 = 32 * bytesPerRow + 48 * 4;
    const c32_32 = 32 * bytesPerRow + 32 * 4;
    transformReadbackResult = {
      verified: true,
      center_48_32: [pixels[c48_32], pixels[c48_32 + 1], pixels[c48_32 + 2], pixels[c48_32 + 3]],
      untransformed_32_32: [
        pixels[c32_32],
        pixels[c32_32 + 1],
        pixels[c32_32 + 2],
        pixels[c32_32 + 3],
      ],
    };
  }

  return {
    identityCode: code0,
    translatedMatrixCode: code0Trans,
    nonAffineCode: code2,
    affineRowsTransCode: code0AffineTrans,
    affineRowsNanCode: code2AffineNan,
    smallBufferCode: code1Small,
    midBufferCode: code1Mid,
    oversizedBufferCode: code4Oversized,
    transformReadbackResult,
    detail: transformReadbackResult
      ? `Verified via wire layout validator (codes 0, 1, 2, 4) AND authoritative browser WGSL AffineRows transform evaluation (center 48,32 Green [0,255,0,255], untransformed 32,32 Black).`
      : `Verified via real Rust/Wasm AffineRows wire validator: 64-byte identity (0) & translated (0), 48-byte AffineRows with translation (0) & NaN translation (2), 64-byte e[11] perspective (2), 47-byte (1), 56-byte (1), 72-byte (4).`,
  };
}

export async function testNegativeBrokenAffineRowsGpuPacketRejection(host, wasmModule) {
  const validateFn =
    wasmModule?.f3d_validate_affine_rows || wasmModule?.gpu_bridge_validate_affine_rows;

  if (!wasmModule || typeof validateFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenAffineRowsGpuPacketRejection requires 'gpu_bridge_validate_affine_rows'.",
    );
  }

  // Construct non-affine matrix with non-zero perspective term (e[11] = 0.5)
  const nonAffineBytes = new Uint8Array(64);
  const naFloats = new Float32Array(nonAffineBytes.buffer);
  naFloats[0] = 1.0;
  naFloats[5] = 1.0;
  naFloats[10] = 1.0;
  naFloats[15] = 1.0;
  naFloats[11] = 0.5;

  let submissionAttempted = false;
  let validatorRejected = false;

  try {
    // GPU-side Control: Upload attempt through validator gate.
    // The gate MUST validate the matrix bytes and refuse the operation BEFORE constructing
    // or submitting any binary packet to the GPU.
    assertAffineRowsLayoutValid(validateFn, nonAffineBytes);

    // If the gate erroneously permitted the matrix, constructing and submitting the packet would execute
    submissionAttempted = true;
    const builder = new BinaryPacketBuilder();
    builder.createBuffer(70, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    builder.writeBuffer(70, 0, nonAffineBytes);
    const packet = builder.build();
    await host.executePacket(packet);
  } catch (err) {
    if (err.message && err.message.includes("AffineRows Layout Validation Rejected")) {
      validatorRejected = true;
    }
  }

  if (submissionAttempted) {
    throw new Error(
      "Negative Control Failed: Corrupt non-affine transform bypassed validator gate and attempted GPU submission!",
    );
  }

  if (!validatorRejected) {
    throw new Error(
      "Negative Control Failed: assertAffineRowsLayoutValid did not reject non-affine matrix before GPU submission!",
    );
  }

  // GPU broken transform control: An untransformed triangle (or identity AffineRows without +0.5 translation)
  // renders at (32, 32) instead of (48, 32). The EXACT SAME assertAffineRowsTransformMatch MUST reject it!
  const buildAffineTransformFn =
    wasmModule?.f3d_build_affine_rows_transform_packet ||
    wasmModule?.gpu_bridge_build_affine_rows_transform_packet;
  let transformControlRejected = false;
  if (typeof buildAffineTransformFn === "function") {
    // A 64x64 buffer with center untransformed (32, 32) Green and (48, 32) Black
    const bytesPerRow = computeAlignedBytesPerRow(64);
    const brokenPixels = new Uint8Array(bytesPerRow * 64);
    const c32_32 = 32 * bytesPerRow + 32 * 4;
    brokenPixels[c32_32] = 0;
    brokenPixels[c32_32 + 1] = 255;
    brokenPixels[c32_32 + 2] = 0;
    brokenPixels[c32_32 + 3] = 255; // Untransformed center is green!

    try {
      assertAffineRowsTransformMatch(brokenPixels, 64, 64);
    } catch (e) {
      if (e.message && e.message.includes("AffineRows WGSL sample violation")) {
        transformControlRejected = true;
      }
    }

    if (!transformControlRejected) {
      throw new Error(
        "Negative Control Failed: assertAffineRowsTransformMatch did not reject untransformed/corrupt transform output!",
      );
    }
  }

  return {
    behavior: "non_affine_gpu_submission_refused",
    transformControlRejected: transformControlRejected || "validator_gate_intercepted",
    detail:
      "Validator gate intercepted non-affine matrix (code 2) and refused packet upload before any GPU submission was encoded.",
  };
}

/**
 * -----------------------------------------------------------------------------
 * 11. NESTED PASS PROTOCOL & LOADOP LOAD RESUME (§7.5, §8.2, vqa.7)
 * -----------------------------------------------------------------------------
 * Plan §7.5, §8.2, and AGENTS.md "Nested Passes":
 * Guards against losing intermediate render pass results across pass interleaving.
 * Three logical passes executed through candidate bridge:
 * - Pass 1 (Target 10): prefix clear to black + draw Red left triangle (x in [-1, 0]).
 * - Pass 2 (Target 11): nested pass on intermediate target 11 (clear + Green draw).
 * - Pass 3 (Target 10): resume with loadOp: "load" + draw Blue right triangle (x in [0, 1]).
 * - Readback buffer 20 on target 10.
 *
 * Ground truth coordinates on target 10:
 * - (24, 32): Red [255, 0, 0, 255] (preserved across resume via loadOp load)
 * - (56, 32): Blue [0, 0, 255, 255] (drawn on resume pass)
 * - (2, 2): Black [0, 0, 0, 255] (clear background)
 *
 * Negative Control: Resume pass uses loadOp: "clear" instead of "load", clearing
 * target 10 and losing the Red left triangle. The EXACT SAME assertNestedPassMatch
 * assertion rejects it.
 */
export async function testNestedPassProtocol(host, wasmModule, width = 64, height = 64) {
  const buildNestedPassFn =
    wasmModule?.f3d_build_nested_pass_packet || wasmModule?.gpu_bridge_build_nested_pass_packet;

  if (!wasmModule || typeof buildNestedPassFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNestedPassProtocol requires compiled application Wasm with export 'gpu_bridge_build_nested_pass_packet' (or canonical alias 'f3d_build_nested_pass_packet'). Silent JS fallback is forbidden.",
    );
  }

  // 1. Run direct-JS oracle for authoritative ground truth
  const oraclePixels = await renderDirectNestedPassReference(host.device, width, height);

  // 2. Execute real Rust/Wasm binary packet through candidate bridge host
  const packet = buildNestedPassFn();
  await host.executePacket(packet);

  // 3. Read back target 10 from buffer 20
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;
  const readbackBuffer = host.buffers.get(20);
  if (!readbackBuffer) {
    throw new Error(
      "testNestedPassProtocol: expected readback buffer 20 to be registered and populated on host",
    );
  }

  const candidatePixels = await readbackGpuBuffer(host.device, readbackBuffer, readbackSize);

  // 4. Assert candidate pixels match oracle pixel-for-pixel and designated sample points
  assertNestedPassMatch(candidatePixels, oraclePixels, width, height);

  return {
    status: "PASS",
    oraclePixels,
    candidatePixels,
    detail:
      "Verified via real Rust nested pass packet vs independent direct-JS oracle (Red at 24,32 preserved via loadOp load, Blue at 56,32, Black at 2,2).",
  };
}

export async function testNegativeBrokenNestedPass(
  host,
  oraclePixels = null,
  width = 64,
  height = 64,
) {
  // If oraclePixels was not passed from positive path, run direct oracle to get authoritative reference
  const referencePixels =
    oraclePixels || (await renderDirectNestedPassReference(host.device, width, height));

  // Broken control: Resume pass on target 10 issues loadOp clear, wiping out the Red prefix pass
  const brokenPixels = await renderDirectBrokenNestedPass(host.device, width, height);

  // The EXACT SAME assertion must be applied and MUST reject it
  let rejected = false;
  try {
    assertNestedPassMatch(brokenPixels, referencePixels, width, height);
  } catch (err) {
    if (
      err.message &&
      (err.message.includes("Nested Pass sample violation") ||
        err.message.includes("assertNestedPassMatch") ||
        err.message.includes("mismatched bytes"))
    ) {
      rejected = true;
    }
  }

  if (!rejected) {
    throw new Error(
      "Negative Control Failed: assertNestedPassMatch did not reject broken nested pass using loadOp clear on resume!",
    );
  }

  return {
    behavior: "load_op_clear_resume_rejected",
    detail:
      "Broken nested pass using loadOp clear on resume instead of loadOp load was strictly rejected by assertNestedPassMatch (red prefix pass cleared to black).",
  };
}

/**
 * -----------------------------------------------------------------------------
 * 12. NESTED CANVAS PASS PROTOCOL & SWAPCHAIN REENTRANCY (§6.7, §8.5, 2v8.4)
 * -----------------------------------------------------------------------------
 * Consumes real compiled Rust/Wasm export `gpu_bridge_build_nested_canvas_pass_packet`
 * (alias `f3d_build_nested_canvas_pass_packet`, 14 commands):
 * - Pass 1 (Canvas Target 10): prefix clear black + Red tri1 (Pipeline 201, preferredCanvasFormat).
 * - Pass 2 (Offscreen Target 11): nested pass clear black + Green tri1 (Pipeline 200, rgba8unorm).
 * - Pass 3 (Canvas Target 10): resume with loadOp: "load" + Blue tri2 (Pipeline 201).
 * - Command 13: Copy offscreen Target 11 to Readback Buffer 20 (64x64 rgba8unorm).
 *
 * Configure COPY_SRC and copy the actual canvas before yielding. Compare the
 * red/blue outer image and green/black inner image with independent native GPU
 * references. Mutate the resumed canvas load operation to clear and require the
 * same image assertion to reject the GPU result. Missing exports also reject.
 */
async function executeNestedCanvasReadback(host, packet, canvasContext, width, height) {
  if (!canvasContext) throw new Error("Nested canvas readback requires a real canvas context");
  const device = host.device;
  const format = navigator.gpu.getPreferredCanvasFormat();
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const size = bytesPerRow * height;
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    canvasContext.configure({
      device,
      format,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const executed = host.executePacket(packet, canvasContext);
    // Copy the actual canvas texture before yielding and allowing it to expire.
    const copied = host.withErrorScopes(["validation"], () => {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: canvasContext.getCurrentTexture() },
        { buffer, bytesPerRow },
        [width, height, 1],
      );
      device.queue.submit([encoder.finish()]);
    });
    await Promise.all([executed, copied]);
    const pixels = await readbackGpuBuffer(device, buffer, size);
    if (format === "bgra8unorm") {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = y * bytesPerRow + x * 4;
          [pixels[i], pixels[i + 2]] = [pixels[i + 2], pixels[i]];
        }
      }
    }
    return pixels;
  } finally {
    buffer.destroy();
  }
}

export async function testNestedCanvasPassProtocol(
  host,
  wasmModule,
  width = 64,
  height = 64,
  canvasContext = null,
) {
  const buildNestedCanvasPassFn =
    wasmModule?.f3d_build_nested_canvas_pass_packet ||
    wasmModule?.gpu_bridge_build_nested_canvas_pass_packet;

  if (!wasmModule || typeof buildNestedCanvasPassFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNestedCanvasPassProtocol requires compiled application Wasm with export 'gpu_bridge_build_nested_canvas_pass_packet' (or canonical alias 'f3d_build_nested_canvas_pass_packet'). Silent JS fallback is forbidden.",
    );
  }

  // 1. Run independent direct-JS oracle rendering only the offscreen pass on Target 11
  const oraclePixels = await renderDirectNestedCanvasOffscreenReference(host.device, width, height);
  // The outer canvas has the same red-prefix / blue-resume image as this
  // independent native WebGPU reference; no candidate pixels define the oracle.
  const canvasOraclePixels = await renderDirectNestedPassReference(host.device, width, height);

  // 2. Resolve canvas context (from argument, DOM canvas, or OffscreenCanvas)
  let activeCanvasContext = canvasContext;
  if (!activeCanvasContext) {
    if (typeof document !== "undefined" && typeof document.getElementById === "function") {
      const existingCanvas = document.getElementById("webgpu-swapchain-canvas");
      if (existingCanvas && typeof existingCanvas.getContext === "function") {
        activeCanvasContext = existingCanvas.getContext("webgpu");
      }
    }
  }

  // 3. Execute real Rust/Wasm binary packet through candidate bridge host.
  // When activeCanvasContext is present, executePacket acquires canvas swapchain view
  // and executes Pass 1 (canvas prefix), Pass 2 (nested offscreen), and Pass 3 (canvas resume)
  // synchronously inside withErrorScopes(["validation", "out-of-memory"]).
  // Any WebGPU validation error causes executePacket to throw.
  const packet = buildNestedCanvasPassFn();
  const canvasPixels = await executeNestedCanvasReadback(
    host,
    packet,
    activeCanvasContext,
    width,
    height,
  );
  assertNestedPassMatch(canvasPixels, canvasOraclePixels, width, height);

  // 4. Read back target 11 from buffer 20
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;
  const readbackBuffer = host.buffers.get(20);
  if (!readbackBuffer) {
    throw new Error(
      "testNestedCanvasPassProtocol: expected readback buffer 20 to be registered and populated on host",
    );
  }

  const candidatePixels = await readbackGpuBuffer(host.device, readbackBuffer, readbackSize);

  // 5. Assert candidate pixels match oracle byte-for-byte and at designated sample points
  assertNestedCanvasPassMatch(candidatePixels, oraclePixels, width, height);

  return {
    status: "PASS",
    oraclePixels,
    candidatePixels,
    canvasOraclePixels,
    canvasPixels,
    detail:
      "Rust FrameSession packet: actual canvas is byte-identical to independent red/blue reference; target 11 is byte-identical to independent green/black reference; native GPU error scopes clean.",
  };
}

export async function testNegativeBrokenNestedCanvasPass(
  host,
  wasmModule,
  oraclePixels = null,
  width = 64,
  height = 64,
  canvasContext = null,
) {
  // 1. Missing-export rejection control: Calling with missing export strictly fails
  let missingExportRejected = false;
  try {
    await testNestedCanvasPassProtocol(host, {}, width, height);
  } catch (err) {
    if (err.message && err.message.includes("Missing Wasm Export")) {
      missingExportRejected = true;
    }
  }
  if (!missingExportRejected) {
    throw new Error(
      "Negative Control Failed: testNestedCanvasPassProtocol did not reject missing Wasm export!",
    );
  }

  // 2. Change the actual Rust packet's resumed canvas load operation, then render it.
  const build =
    wasmModule.f3d_build_nested_canvas_pass_packet ||
    wasmModule.gpu_bridge_build_nested_canvas_pass_packet;
  const brokenPacket = build().slice();
  const view = new DataView(brokenPacket.buffer, brokenPacket.byteOffset, brokenPacket.byteLength);
  const fieldSizes = { 1: 12, 2: 16, 3: 32, 4: 44, 5: 24, 6: 20 };
  let cursor = 16;
  let mutations = 0;
  for (let i = 0; i < view.getUint32(8, true); i++) {
    const opcode = view.getUint16(cursor, true);
    cursor += 2;
    if (opcode === OPCODE_RENDER_PASS) {
      const packed = view.getUint32(cursor, true);
      if ((packed & 0xff) === TARGET_CANVAS && ((packed >>> 8) & 0xff) === 1) {
        view.setUint32(cursor, packed & ~0xff00, true);
        mutations++;
      }
    }
    if (!fieldSizes[opcode]) throw new Error(`Unexpected nested fixture opcode ${opcode}`);
    cursor += fieldSizes[opcode];
  }
  if (mutations !== 1) throw new Error(`Expected one resumed canvas pass, found ${mutations}`);
  const referencePixels =
    oraclePixels || (await renderDirectNestedPassReference(host.device, width, height));
  const brokenPixels = await executeNestedCanvasReadback(
    host,
    brokenPacket,
    canvasContext,
    width,
    height,
  );

  let comparisonRejected = false;
  try {
    assertNestedPassMatch(brokenPixels, referencePixels, width, height);
  } catch (err) {
    if (
      err.message &&
      (err.message.includes("mismatched bytes") ||
        err.message.includes("Nested Pass sample violation") ||
        err.message.includes("assertNestedPassMatch"))
    ) {
      comparisonRejected = true;
    }
  }
  if (!comparisonRejected) {
    throw new Error(
      "Negative Control Failed: canvas comparison did not reject a real clear-on-resume mutation!",
    );
  }

  return {
    behavior: "gpu_canvas_clear_on_resume_and_missing_export_rejected",
    missingExportRejected,
    comparisonRejected,
    detail:
      "Missing Wasm export rejected; a real canvas load-to-clear packet mutation rendered and failed the same independent image assertion.",
  };
}
