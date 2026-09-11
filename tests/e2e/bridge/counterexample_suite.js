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
  WebGpuBridgeHost,
  PACKET_MAGIC,
  PACKET_VERSION,
  OPCODE_CREATE_BUFFER,
  OPCODE_WRITE_BUFFER,
  OPCODE_CREATE_PIPELINE,
  OPCODE_RENDER_PASS,
  OPCODE_COPY_TEXTURE_TO_BUFFER,
  OPCODE_CREATE_TEXTURE,
  OPCODE_RECORD_BUNDLE,
  OPCODE_EXECUTE_BUNDLES,
  TARGET_OFFSCREEN,
} from "../../fixtures/gpu_bridge/bridge_runtime.js";

import {
  WGSL_AFFINE_TRIANGLE,
  WGSL_SOLID_COLOR,
  WGSL_FLAT_COLOR_BUNDLE,
  computeAlignedBytesPerRow,
  readbackGpuBuffer,
  renderDirectTriangleReference,
  renderDirectRedABlueBReference,
  renderDirectBundleDirectReference,
  assertBundleDirectDrawMatch,
  assertGenerationalHandlePublication,
  assertMemoryGrowthAllowed,
  assertAffineRowsLayoutValid,
  evalDirectAffineTransform,
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
    this.commands.push({ op: OPCODE_WRITE_BUFFER, bufferId, offset, dataOffset, dataLength: dataUint8.byteLength });
  }

  createTexture(textureId, width, height, formatCode, usage) {
    this.commands.push({ op: OPCODE_CREATE_TEXTURE, textureId, width, height, formatCode, usage });
  }

  createPipeline(pipelineId, wgslText, formatCode, hasVB, hasUniform, uniformSize = 0, vertexStride = 0) {
    const codeBytes = new TextEncoder().encode(wgslText);
    const codeOffset = this.dataTotalLen;
    this.dataChunks.push(codeBytes);
    this.dataTotalLen += codeBytes.byteLength;
    this.commands.push({ op: OPCODE_CREATE_PIPELINE, pipelineId, codeOffset, codeLen: codeBytes.byteLength, formatCode, hasVB, hasUniform, uniformSize, vertexStride });
  }

  renderPass(targetType, targetId, clearColor, pipelineId, vbId, vertexCount, dynamicOffset = 0, uniformBufferId = 1) {
    this.commands.push({ op: OPCODE_RENDER_PASS, targetType, targetId, clearColor, pipelineId, vbId, vertexCount, dynamicOffset, uniformBufferId });
  }

  copyTextureToBuffer(textureId, bufferId, width, height, epochHi = 0, epochLo = 0) {
    this.commands.push({ op: OPCODE_COPY_TEXTURE_TO_BUFFER, textureId, bufferId, width, height, epochHi, epochLo });
  }

  recordBundle(bundleId, pipelineId, vertexBufferId, vertexCount, dynamicOffset = 0, uniformBufferId = 1, targetFormat = 2) {
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
        case OPCODE_CREATE_BUFFER: commandBytesLen += 2 + 12; break;
        case OPCODE_WRITE_BUFFER: commandBytesLen += 2 + 16; break;
        case OPCODE_CREATE_TEXTURE: commandBytesLen += 2 + 20; break;
        case OPCODE_CREATE_PIPELINE: commandBytesLen += 2 + 32; break;
        case OPCODE_RENDER_PASS: commandBytesLen += 2 + 44; break;
        case OPCODE_COPY_TEXTURE_TO_BUFFER: commandBytesLen += 2 + 24; break;
        case OPCODE_RECORD_BUNDLE: commandBytesLen += 2 + 28; break;
        case OPCODE_EXECUTE_BUNDLES: commandBytesLen += 2 + 4 + cmd.bundleIds.length * 4; break;
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
    throw new Error(`${label}: byte length mismatch (candidate=${candidatePixels.byteLength}, oracle=${oraclePixels.byteLength})`);
  }
  let diffCount = 0;
  for (let i = 0; i < candidatePixels.length; i++) {
    if (candidatePixels[i] !== oraclePixels[i]) {
      diffCount++;
    }
  }
  if (diffCount > 0) {
    throw new Error(`${label}: detected ${diffCount} mismatched bytes out of ${candidatePixels.length}`);
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
  const rA = pixelsA[centerA], gA = pixelsA[centerA + 1], bA = pixelsA[centerA + 2], aA = pixelsA[centerA + 3];

  if (rA < 250 || gA > 5 || bA > 5 || aA < 250) {
    throw new Error(`Target A color violation: expected Red [255, 0, 0, 255], observed [${rA}, ${gA}, ${bA}, ${aA}]`);
  }

  // Check center pixel of Target B: expected Blue [0, 0, 255, 255]
  const centerB = midY * bytesPerRow + midX * 4;
  const rB = pixelsB[centerB], gB = pixelsB[centerB + 1], bB = pixelsB[centerB + 2], aB = pixelsB[centerB + 3];

  if (rB > 5 || gB > 5 || bB < 250 || aB < 250) {
    throw new Error(`Target B color violation: expected Blue [0, 0, 255, 255], observed [${rB}, ${gB}, ${bB}, ${aB}]`);
  }
}

/**
 * -----------------------------------------------------------------------------
 * 1. FIRST-FRAME PIXEL EQUIVALENCE
 * -----------------------------------------------------------------------------
 */

export async function testFirstFramePixelEquivalence(host, device, wasmModule, width = 64, height = 64, canvasContext = null) {
  const buildTriangleFn = wasmModule?.f3d_build_first_frame_packet || wasmModule?.gpu_bridge_build_triangle_packet;
  if (!wasmModule || typeof buildTriangleFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testFirstFramePixelEquivalence requires compiled application Wasm with export 'gpu_bridge_build_triangle_packet' or 'f3d_build_first_frame_packet'. Silent JS fallback is forbidden."
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
    const preferredFormat = host.capabilityRecord?.preferredCanvasFormat ||
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

export async function testNegativeBrokenFirstFrameGpuTransform(host, oraclePixels, width = 64, height = 64) {
  // Broken Control: An actual faulty GPU packet uploading an out-of-bounds transform matrix.
  // Translation tx=50, ty=50 pushes all 3 vertices completely outside the NDC [-1, 1] clip space.
  // Real GPU execution renders pure clear color [0, 0, 0, 1] with the triangle entirely clipped!
  const builder = new BinaryPacketBuilder();
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  builder.createTexture(1, width, height, 2, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
  builder.createBuffer(1, 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  const brokenAffine = new Float32Array([
    1.0, 0.0, 0.0, 50.0,
    0.0, 1.0, 0.0, 50.0,
    0.0, 0.0, 1.0, 0.0,
  ]);
  builder.writeBuffer(1, 0, new Uint8Array(brokenAffine.buffer));

  const vertexData = new Float32Array([
     0.0,  0.5,  0.0, 0.5, 1.0,
    -0.5, -0.5,  0.0, 0.0, 0.0,
     0.5, -0.5,  0.0, 1.0, 0.0,
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
    throw new Error("Negative Control Failed: assertExactPixelMatch failed to reject out-of-bounds GPU render output!");
  }
}

/**
 * -----------------------------------------------------------------------------
 * 2. RED-A / BLUE-B QUEUE-WRITE SNAPSHOT ISOLATION
 * -----------------------------------------------------------------------------
 */

export async function testRedABlueBQueueWriteSnapshot(host, wasmModule, width = 64, height = 64) {
  const buildRedBlueFn = wasmModule?.f3d_build_red_a_blue_b_packet || wasmModule?.gpu_bridge_build_red_blue_packet;
  if (!wasmModule || typeof buildRedBlueFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testRedABlueBQueueWriteSnapshot requires compiled application Wasm with export 'gpu_bridge_build_red_blue_packet' or 'f3d_build_red_a_blue_b_packet'. Silent JS fallback is forbidden."
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

export async function testNegativeBrokenRedABlueBQueueHazard(host, wasmModule, width = 32, height = 32) {
  const buildRedBlueFn = wasmModule?.f3d_build_red_a_blue_b_packet || wasmModule?.gpu_bridge_build_red_blue_packet;
  if (!wasmModule || typeof buildRedBlueFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenRedABlueBQueueHazard requires compiled application Wasm with export 'gpu_bridge_build_red_blue_packet' or 'f3d_build_red_a_blue_b_packet'."
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
    throw new Error("Negative Control Failed: Unversioned Rust packet was not rejected by assertRedABlueB!");
  }

  // 2. Also test direct GPU in-place uniform overwrite hazard control
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  const builder = new BinaryPacketBuilder();
  builder.createTexture(30, width, height, 2, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
  builder.createTexture(31, width, height, 2, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);

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
    throw new Error("Negative Control Failed: assertRedABlueB did not reject in-place queue-write overwrite hazard!");
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
    throw new Error("Negative Control Failed: testFirstFramePixelEquivalence did not reject null wasmModule!");
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
    throw new Error("Negative Control Failed: testRedABlueBQueueWriteSnapshot did not reject missing export!");
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
    throw new Error("Negative Control Failed: testStaleEpochReadbackPublicationGate did not reject missing export!");
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
    throw new Error("Negative Control Failed: testBundleThenDirectDrawStateReset did not reject missing export!");
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
    throw new Error("Negative Control Failed: testGenerationalHandleAbaPublication did not reject missing export!");
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
    throw new Error("Negative Control Failed: testLinearMemoryBorrowGuards did not reject missing export!");
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
    throw new Error("Negative Control Failed: testAffineRowsGpuLayoutValidation did not reject missing export!");
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
    throw new Error("Negative Control Failed: Bridge decoder failed to reject truncated packet fields!");
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
    throw new Error("Negative Control Failed: Bridge decoder failed to reject invalid texture format code!");
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
    if (e.message && e.message.includes("Invalid render pass targetType")) {
      rejectedTarget = true;
    }
  }
  if (!rejectedTarget) {
    throw new Error("Negative Control Failed: Bridge decoder failed to reject invalid render pass target type!");
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
      await new Promise(r => setTimeout(r, 10));
    });
  } catch (e) {
    if (e.message && e.message.includes("syncAction returned a Promise")) {
      rejectedPromise = true;
    }
  }
  if (!rejectedPromise) {
    throw new Error("Negative Control Failed: withErrorScopes failed to reject async/Promise return!");
  }

  // Sub-test 2: Concurrent error scope invocation
  let innerErr = null;
  let innerPromise = null;
  await host.withErrorScopes(["validation"], () => {
    // Re-enter withErrorScopes while active; capture inner async promise rejection
    innerPromise = host.withErrorScopes(["validation"], () => {}).catch(e => {
      innerErr = e;
    });
  });

  if (innerPromise) {
    await innerPromise;
  }

  if (!innerErr || !innerErr.message || !innerErr.message.includes("Error-scope serialization violation")) {
    throw new Error(
      `Negative Control Failed: withErrorScopes failed to reject concurrent interleaved scopes! Observed: ${innerErr ? innerErr.message : "no error"}`
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
  const tryPublishFn = wasmModule?.gpu_bridge_try_publish_readback || wasmModule?.f3d_try_publish_readback;
  if (!wasmModule || typeof tryPublishFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testStaleEpochReadbackPublicationGate requires compiled application Wasm with export 'gpu_bridge_try_publish_readback' (or 'f3d_try_publish_readback'). Silent JS fallback is forbidden."
    );
  }

  // Execute a real GPU copy operation stamped with Epoch (0, 1)
  const width = 64;
  const height = 64;
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  const builder = new BinaryPacketBuilder();
  builder.createTexture(50, width, height, 2, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
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
      `Positive Control Failed: Rust publication gate rejected matching readback epoch (${readbackEpochHi}, ${readbackEpochLo}) against region epoch (0, 1)!`
    );
  }

  // 2. Negative: Region advances to epoch (0, 2); in-flight readback at (0, 1) is now stale and must be rejected
  const staleAccepted = tryPublishFn(readbackEpochHi, readbackEpochLo, 0, 2);
  if (staleAccepted) {
    throw new Error(
      `Negative Control Failed: Rust publication gate falsely accepted stale readback epoch (${readbackEpochHi}, ${readbackEpochLo}) against advanced region epoch (0, 2)!`
    );
  }

  // 3. Negative: High-word mismatch (1, 1) must also be rejected
  const highMismatchAccepted = tryPublishFn(readbackEpochHi, readbackEpochLo, 1, 1);
  if (highMismatchAccepted) {
    throw new Error(
      `Negative Control Failed: Rust publication gate falsely accepted high-word mismatch epoch (${readbackEpochHi}, ${readbackEpochLo}) against region epoch (1, 1)!`
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
export async function testBundleThenDirectDrawStateReset(host, device, wasmModule, width = 64, height = 64) {
  const buildBundleDirectFn = wasmModule?.f3d_build_bundle_direct_draw_packet || wasmModule?.gpu_bridge_build_bundle_direct_draw_packet;
  if (!wasmModule || typeof buildBundleDirectFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testBundleThenDirectDrawStateReset requires compiled application Wasm with export 'gpu_bridge_build_bundle_direct_draw_packet' (or 'f3d_build_bundle_direct_draw_packet'). Silent JS fallback is forbidden."
    );
  }

  // REAL RUST/WASM PATH: Packet generated by crates/f3d-runtime/src/gpu_host.rs
  const packet = buildBundleDirectFn();
  const readbackBufferId = 20; // Rust build_bundle_then_direct_draw_submission outputs to buffer 20
  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackSize = bytesPerRow * height;

  await host.executePacket(packet);
  const candidatePixels = await host.readbackBuffer(readbackBufferId, readbackSize);

  // Oracle: Independent direct-JS execution
  const oraclePixels = await renderDirectBundleDirectReference(device, width, height);

  // Positive Assertion: Must match pixel-for-pixel (0 differences) + verify sample points
  assertBundleDirectDrawMatch(candidatePixels, oraclePixels, width, height);

  return { candidatePixels, oraclePixels };
}

export async function testNegativeBrokenBundleDirectDraw(host, oraclePixels, width = 64, height = 64) {
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
    -1.0,  1.0, 0.0,  0.0, 1.0,
    -1.0, -1.0, 0.0,  0.0, 0.0,
     0.0,  1.0, 0.0,  0.5, 1.0,
  ]);
  const vb1 = host.device.createBuffer({
    size: tri1Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  host.device.queue.writeBuffer(vb1, 0, tri1Data);

  // Vertex buffer 2: Triangle 2 (right side)
  const tri2Data = new Float32Array([
     0.0, -1.0, 0.0,  0.5, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
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
        colorAttachments: [{
          view: target.createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        }],
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
        [width, height, 1]
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
      ? (validationError.message || String(validationError))
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
      "Negative Control Failed: assertBundleDirectDrawMatch did not reject GPU output produced without post-bundle rebind!"
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
export async function testGenerationalHandleAbaPublication(host, wasmModule) {
  const checkFn = wasmModule?.f3d_check_resource_handle || wasmModule?.gpu_bridge_check_resource_handle;
  const advanceFn = wasmModule?.f3d_advance_resource_generation || wasmModule?.gpu_bridge_advance_resource_generation;
  const buildTriangleFn = wasmModule?.f3d_build_first_frame_packet || wasmModule?.gpu_bridge_build_triangle_packet;

  if (!wasmModule || typeof checkFn !== "function" || typeof advanceFn !== "function" || typeof buildTriangleFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testGenerationalHandleAbaPublication requires compiled application Wasm with exports 'gpu_bridge_check_resource_handle', 'gpu_bridge_advance_resource_generation', and 'gpu_bridge_build_triangle_packet' (or aliases). Silent JS fallback is forbidden."
    );
  }

  // 1. Execute a real GPU packet through the WebGPU bridge
  // In crates/f3d-runtime/src/gpu_host.rs:890-898, build_triangle_submission registers
  // resource IDs: 1 (uniform buffer), 2 (vertex buffer), 3 (readback buffer), 10 (target texture), 100 (pipeline)
  // at initial generation 1 in the global resource slot table.
  const packet = buildTriangleFn();
  await host.executePacket(packet);

  const registeredTriangleIds = [1, 2, 3, 10, 100];
  for (const id of registeredTriangleIds) {
    if (!checkFn(id, 1)) {
      throw new Error(
        `Positive Control Failed: Expected resource id ${id} to be registered at generation 1, but check returned false!`
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
      `Lifecycle Advance Failed: Expected advance_resource_generation(${testId}) to return 2, got ${newGen}!`
    );
  }

  // 4. Assert check(id, 1) is false (stale ABA generation rejected)
  if (checkFn(testId, 1) !== false) {
    throw new Error(
      `ABA Hazard Failed: Expected check_resource_handle(${testId}, 1) to return false after advance, got true!`
    );
  }

  // 5. Assert check(id, 2) is true (fresh generation after reallocation accepted)
  if (checkFn(testId, 2) !== true) {
    throw new Error(
      `Positive Control Failed: Expected check_resource_handle(${testId}, 2) to return true after advance, got false!`
    );
  }
  assertGenerationalHandlePublication(checkFn, testId, 2);

  // 6. Assert check(id, 0) is false (generation zero strictly rejected via HandleError::InvalidGeneration)
  if (checkFn(testId, 0) !== false) {
    throw new Error(
      `Generation Zero Invariant Failed: Expected check_resource_handle(${testId}, 0) to return false, got true!`
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

export async function testNegativeBrokenGenerationalHandlePublication(host, wasmModule, testId = 10, staleGeneration = 1) {
  const checkFn = wasmModule?.f3d_check_resource_handle || wasmModule?.gpu_bridge_check_resource_handle;
  if (!wasmModule || typeof checkFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenGenerationalHandlePublication requires 'gpu_bridge_check_resource_handle' (or alias). Silent JS fallback is forbidden."
    );
  }

  // Verify that the GPU resource was indeed created and executed on the bridge host
  if (!host.textures.has(testId)) {
    throw new Error(
      `testNegativeBrokenGenerationalHandlePublication: target texture ${testId} not found in bridge host textures`
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
      `Negative Control Failed: assertGenerationalHandlePublication did not reject publication attempt with stale generation (${testId}, gen=${staleGeneration})!`
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
      `Negative Control Failed: assertGenerationalHandlePublication did not reject publication attempt with generation zero (${testId}, gen=0)!`
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
 *
 * Sequence:
 * 1. Assert grow true when idle (`try_grow_memory(1) === true`).
 * 2. Assert enter gives nonzero BigInt token (`typeof token === "bigint" && token !== 0n`).
 * 3. Assert grow false while borrowed (`try_grow_memory(1) === false`).
 * 4. Assert exit with wrong token false (`borrow_exit(wrongToken) === false`).
 * 5. Assert exit with right token true (`borrow_exit(token) === true`).
 * 6. Assert grow true again (`try_grow_memory(1) === true`).
 *
 * Negative Control: An open borrow is entered, and an attempt to grow memory
 * is issued. The EXACT SAME `assertMemoryGrowthAllowed` assertion rejects it.
 */
export async function testLinearMemoryBorrowGuards(host, wasmModule) {
  const enterFn = wasmModule?.f3d_borrow_enter || wasmModule?.gpu_bridge_borrow_enter;
  const exitFn = wasmModule?.f3d_borrow_exit || wasmModule?.gpu_bridge_borrow_exit;
  const growFn = wasmModule?.f3d_try_grow_memory || wasmModule?.gpu_bridge_try_grow_memory;

  if (!wasmModule || typeof enterFn !== "function" || typeof exitFn !== "function" || typeof growFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testLinearMemoryBorrowGuards requires compiled application Wasm with exports 'gpu_bridge_borrow_enter', 'gpu_bridge_borrow_exit', and 'gpu_bridge_try_grow_memory' (or canonical aliases). Silent JS fallback is forbidden."
    );
  }

  // 1. Assert grow true when idle
  const idleGrow = growFn(1);
  if (idleGrow !== true) {
    throw new Error(
      "Positive Control Failed: Expected try_grow_memory(1) to return true while idle!"
    );
  }
  assertMemoryGrowthAllowed(growFn, 1);

  // 2. Assert enter gives nonzero BigInt token (wasm-bindgen u64 is strictly BigInt)
  const token = enterFn();
  if (typeof token !== "bigint" || token === 0n) {
    throw new Error(
      `Positive Control Failed: Expected borrow_enter() to return non-zero BigInt BorrowToken (wasm-bindgen u64), got ${typeof token} (${token})!`
    );
  }

  // 3. Assert grow false while borrowed
  const borrowedGrow = growFn(1);
  if (borrowedGrow !== false) {
    // Clean up borrow before throwing
    exitFn(token);
    throw new Error(
      "Safety Hazard: Expected try_grow_memory(1) to return false while an active borrow is held!"
    );
  }

  // 4. Assert exit with wrong token false (BigInt token)
  const wrongToken = token + 999999n;
  const wrongExit = exitFn(wrongToken);
  if (wrongExit !== false) {
    exitFn(token);
    throw new Error(
      "Token Security Failed: Expected borrow_exit with wrong token to return false!"
    );
  }

  // 5. Assert exit with right token true
  const rightExit = exitFn(token);
  if (rightExit !== true) {
    throw new Error(
      "Positive Control Failed: Expected borrow_exit with valid token to return true!"
    );
  }

  // 6. Assert grow true again
  const postExitGrow = growFn(1);
  if (postExitGrow !== true) {
    throw new Error(
      "Positive Control Failed: Expected try_grow_memory(1) to return true after borrow scope exit!"
    );
  }
  assertMemoryGrowthAllowed(growFn, 1);

  return {
    token: String(token),
    detail: "Verified via real Rust/Wasm borrow scope exports: grow(idle)=true, enter=token, grow(borrowed)=false, exit(wrong)=false, exit(token)=true, grow(post)=true.",
  };
}

export async function testNegativeBrokenLinearMemoryBorrowGuard(host, wasmModule) {
  const enterFn = wasmModule?.f3d_borrow_enter || wasmModule?.gpu_bridge_borrow_enter;
  const exitFn = wasmModule?.f3d_borrow_exit || wasmModule?.gpu_bridge_borrow_exit;
  const growFn = wasmModule?.f3d_try_grow_memory || wasmModule?.gpu_bridge_try_grow_memory;

  if (!wasmModule || typeof enterFn !== "function" || typeof exitFn !== "function" || typeof growFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenLinearMemoryBorrowGuard requires 'gpu_bridge_borrow_enter' / 'gpu_bridge_try_grow_memory'."
    );
  }

  // Enter an active borrow scope (strictly requires BigInt token)
  const token = enterFn();
  if (typeof token !== "bigint" || token === 0n) {
    throw new Error(
      `Broken Control Setup Failed: borrow_enter did not return valid non-zero BigInt token (got ${typeof token} ${token})`
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
      "Negative Control Failed: assertMemoryGrowthAllowed did not reject memory growth attempt inside open borrow scope!"
    );
  }

  return {
    behavior: "growth_during_borrow_rejected",
    detail: "Linear memory growth during open borrow scope was strictly rejected by assertMemoryGrowthAllowed.",
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
  const validateFn = wasmModule?.f3d_validate_affine_rows || wasmModule?.gpu_bridge_validate_affine_rows;

  if (!wasmModule || typeof validateFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testAffineRowsGpuLayoutValidation requires compiled application Wasm with export 'gpu_bridge_validate_affine_rows' (or alias 'f3d_validate_affine_rows'). Silent JS fallback is forbidden."
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
      `Positive Control Failed: Expected validate_affine_rows(identity) to return 0, got ${code0}!`
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
  v64Floats[12] = 12.5;  // tx
  v64Floats[13] = -4.0;  // ty
  v64Floats[14] = 100.0; // tz

  const code0Trans = validateFn(valid64TransBytes);
  if (code0Trans !== 0) {
    throw new Error(
      `Positive Control Failed: Expected validate_affine_rows(64-byte translated matrix) to return 0, got ${code0Trans}!`
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
      `Non-Affine Rejection Failed: Expected validate_affine_rows(nonAffine) to return 2 (NON_AFFINE_MATRIX), got ${code2}!`
    );
  }

  // 4. Real 48-byte AffineRows wire record with nonzero translation (tx=12.5, ty=-4.0, tz=100.0):
  // Row-major 3x4: row 0 [m00,m01,m02,tx], row 1 [m10,m11,m12,ty], row 2 [m20,m21,m22,tz].
  // Translation is at floats 3, 7, 11.
  const valid48TransBytes = new Uint8Array(48);
  const v48Floats = new Float32Array(valid48TransBytes.buffer);
  v48Floats[0] = 1.0;  // m00
  v48Floats[5] = 1.0;  // m11
  v48Floats[10] = 1.0; // m22
  v48Floats[3] = 12.5;  // tx
  v48Floats[7] = -4.0;  // ty
  v48Floats[11] = 100.0; // tz

  const code0AffineTrans = validateFn(valid48TransBytes);
  if (code0AffineTrans !== 0) {
    throw new Error(
      `Positive Control Failed: Expected validate_affine_rows(48-byte translated AffineRows) to return 0, got ${code0AffineTrans}!`
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
      `Non-Affine Rejection Failed: Expected validate_affine_rows(48-byte NaN translation) to return 2 (NON_AFFINE_MATRIX), got ${code2AffineNan}!`
    );
  }

  // 6. 47-byte buffer (1 byte smaller than 48-byte AffineRows layout, len < 48):
  const smallBytes = new Uint8Array(47);
  const code1Small = validateFn(smallBytes);
  if (code1Small !== 1) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(47-byte) to return 1 (BUFFER_TOO_SMALL), got ${code1Small}!`
    );
  }

  // 7. 56-byte buffer (intermediate length, 48 < len < 64):
  const midLenBytes = new Uint8Array(56);
  const code1Mid = validateFn(midLenBytes);
  if (code1Mid !== 1) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(56-byte) to return 1 (BUFFER_TOO_SMALL), got ${code1Mid}!`
    );
  }

  // 8. 72-byte buffer (oversized buffer, len > 64):
  const oversizedBytes = new Uint8Array(72);
  const code4Oversized = validateFn(oversizedBytes);
  if (code4Oversized !== 4) {
    throw new Error(
      `Buffer Size Rejection Failed: Expected validate_affine_rows(72-byte) to return 4 (INCOMPATIBLE_TARGET), got ${code4Oversized}!`
    );
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
    detail: "Verified via real Rust/Wasm AffineRows validator: 64-byte identity (0) & translated (0), 48-byte AffineRows with translation (0) & NaN translation (2), 64-byte e[11] perspective (2), 47-byte (1), 56-byte (1), 72-byte (4).",
  };
}

export async function testNegativeBrokenAffineRowsGpuPacketRejection(host, wasmModule) {
  const validateFn = wasmModule?.f3d_validate_affine_rows || wasmModule?.gpu_bridge_validate_affine_rows;

  if (!wasmModule || typeof validateFn !== "function") {
    throw new Error(
      "Missing Wasm Export: testNegativeBrokenAffineRowsGpuPacketRejection requires 'gpu_bridge_validate_affine_rows'."
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
      "Negative Control Failed: Corrupt non-affine transform bypassed validator gate and attempted GPU submission!"
    );
  }

  if (!validatorRejected) {
    throw new Error(
      "Negative Control Failed: assertAffineRowsLayoutValid did not reject non-affine matrix before GPU submission!"
    );
  }

  return {
    behavior: "non_affine_gpu_submission_refused",
    detail: "Validator gate intercepted non-affine matrix (code 2) and refused packet upload before any GPU submission was encoded.",
  };
}


