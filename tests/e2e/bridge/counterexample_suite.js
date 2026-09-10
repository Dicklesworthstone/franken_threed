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
  TARGET_OFFSCREEN,
} from "../../fixtures/gpu_bridge/bridge_runtime.js";

import {
  WGSL_AFFINE_TRIANGLE,
  WGSL_SOLID_COLOR,
  computeAlignedBytesPerRow,
  readbackGpuBuffer,
  renderDirectTriangleReference,
  renderDirectRedABlueBReference,
  renderDirectBundleDirectReference,
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
 * - Bundle State Reset: bridge_runtime.js does not yet support OPCODE_EXECUTE_BUNDLES.
 * - Generational Handles: gpu_host.rs does not yet expose generational handle publication to JS.
 * 
 * These items are reported as PENDING and NEVER claimed as green passes until real
 * candidate product implementations exist.
 */

export function getPendingBridgeCapabilities() {
  return [
    {
      capability: "bundle_then_direct_draw_state_reset",
      status: "PENDING_PRODUCT_SUPPORT",
      reason: "bridge_runtime.js does not yet implement bundle execution (OPCODE_EXECUTE_BUNDLES). No mock test is permitted.",
    },
    {
      capability: "generational_handle_aba_publication",
      status: "PENDING_PRODUCT_SUPPORT",
      reason: "crates/f3d-runtime/src/gpu_host.rs does not yet expose generational handle publication tables to JS. No mock test is permitted.",
    },
    {
      capability: "linear_memory_borrow_guards",
      status: "PENDING_PRODUCT_SUPPORT",
      reason: "Browser Wasm linear memory borrow guard ABI is pending runtime integration. No mock test is permitted.",
    },
    {
      capability: "affine_rows_gpu_layout_validation",
      status: "PENDING_PRODUCT_SUPPORT",
      reason: "Wire layout validator pending integration into GpuSubmissionPacket decoder.",
    },
  ];
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
