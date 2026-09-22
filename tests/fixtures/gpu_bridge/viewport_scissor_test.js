import {
  OPCODE_SET_SCISSOR_RECT,
  OPCODE_SET_VIEWPORT,
  PACKET_MAGIC,
  PACKET_VERSION,
  WebGpuBridgeHost,
} from "./bridge_runtime.js";

/**
 * Direct WebGPU oracle reference rendering outer, nested, and resumed passes
 * with independent setViewport and setScissorRect calls for pixel-identical comparison.
 */
async function renderDirectViewportScissorReference(device, width, height) {
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const target10 = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const target11 = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback10 = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const readback11 = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const shaderModule = device.createShaderModule({
    code: `
      struct ColorUniform {
        color: vec4<f32>,
      };
      @group(0) @binding(0)
      var<uniform> u: ColorUniform;

      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) uv: vec2<f32>,
      };

      @vertex
      fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
        return vec4<f32>(in.position, 1.0);
      }

      @fragment
      fn fs_main() -> @location(0) vec4<f32> {
        return u.color;
      }
    `,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 16,
        },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
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
    primitive: {
      topology: "triangle-list",
    },
  });

  // Tri1: covers x in [-1, 0] (left half)
  const vb1Data = new Float32Array([
    -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0,
  ]);
  const vb1 = device.createBuffer({
    size: vb1Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb1, 0, vb1Data);

  // Tri2: covers x in [0, 1] (right half)
  const vb2Data = new Float32Array([
    0.0, -1.0, 0.0, 0.5, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
  ]);
  const vb2 = device.createBuffer({
    size: vb2Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb2, 0, vb2Data);

  // Uniform buffer: 768 bytes
  // Slot 0 (offset 0): Red [1, 0, 0, 1]
  // Slot 1 (offset 256): Blue [0, 0, 1, 1]
  // Slot 2 (offset 512): Green [0, 1, 0, 1]
  const uniformData = new Float32Array(768 / 4);
  uniformData[0] = 1.0;
  uniformData[1] = 0.0;
  uniformData[2] = 0.0;
  uniformData[3] = 1.0;
  uniformData[64] = 0.0;
  uniformData[65] = 0.0;
  uniformData[66] = 1.0;
  uniformData[67] = 1.0;
  uniformData[128] = 0.0;
  uniformData[129] = 1.0;
  uniformData[130] = 0.0;
  uniformData[131] = 1.0;

  const uniformBuf = device.createBuffer({
    size: 768,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, uniformData);

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuf,
          offset: 0,
          size: 16,
        },
      },
    ],
  });

  const encoder = device.createCommandEncoder();

  // Pass 1: Outer prefix on Target 10 (clear to black)
  const pass1 = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target10.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass1.setPipeline(pipeline);
  pass1.setBindGroup(0, bindGroup, [0]); // dynamic offset 0 = Red
  pass1.setVertexBuffer(0, vb1);
  pass1.setViewport(0, 0, 64, 64, 0, 1);
  pass1.setScissorRect(0, 0, 32, 64);
  pass1.draw(3, 1, 0, 0);
  pass1.end();

  // Pass 2: Nested inner pass on Target 11 (clear to black)
  const pass2 = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target11.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass2.setPipeline(pipeline);
  pass2.setBindGroup(0, bindGroup, [512]); // dynamic offset 512 = Green
  pass2.setVertexBuffer(0, vb1);
  pass2.setViewport(16, 16, 32, 32, 0, 1);
  pass2.setScissorRect(16, 16, 32, 32);
  pass2.draw(3, 1, 0, 0);
  pass2.end();

  // Pass 3: Outer resumed pass on Target 10 (LoadOp::Load)
  const pass3 = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target10.createView(),
        loadOp: "load",
        storeOp: "store",
      },
    ],
  });
  pass3.setPipeline(pipeline);
  pass3.setBindGroup(0, bindGroup, [256]); // dynamic offset 256 = Blue
  pass3.setVertexBuffer(0, vb2);
  pass3.setViewport(0, 0, 64, 64, 0, 1);
  pass3.setScissorRect(32, 0, 32, 64);
  pass3.draw(3, 1, 0, 0);
  pass3.end();

  encoder.copyTextureToBuffer(
    { texture: target10 },
    { buffer: readback10, bytesPerRow, rowsPerImage: height },
    [width, height, 1],
  );
  encoder.copyTextureToBuffer(
    { texture: target11 },
    { buffer: readback11, bytesPerRow, rowsPerImage: height },
    [width, height, 1],
  );

  device.queue.submit([encoder.finish()]);

  await Promise.all([readback10.mapAsync(GPUMapMode.READ), readback11.mapAsync(GPUMapMode.READ)]);
  const copy10 = new Uint8Array(readback10.getMappedRange(0, bytesPerRow * height).slice(0));
  const copy11 = new Uint8Array(readback11.getMappedRange(0, bytesPerRow * height).slice(0));
  readback10.unmap();
  readback11.unmap();

  target10.destroy();
  target11.destroy();
  readback10.destroy();
  readback11.destroy();
  uniformBuf.destroy();
  vb1.destroy();
  vb2.destroy();

  return { target10Pixels: copy10, target11Pixels: copy11 };
}

/**
 * Builds an invalid packet that executes SetViewport or SetScissorRect outside an active pass.
 */
function buildOrphanViewportPacket(opcode) {
  const bytes = new Uint8Array(16 + 2 + 24);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, PACKET_MAGIC, true);
  view.setUint16(4, PACKET_VERSION, true);
  view.setUint16(6, 0, true);
  view.setUint32(8, 1, true);
  view.setUint32(12, 0, true);

  view.setUint16(16, opcode, true);
  return bytes;
}

/**
 * Mutates a specified scissor rectangle in the actual Rust packet to prove test sensitivity.
 * - scissorIndex 1: inner Pass 2 scissor (Target 11)
 * - scissorIndex 2: resumed Pass 3 scissor (Target 10)
 */
function mutateRustScissorPacket(originalBytes, scissorIndex, newWidth = 0) {
  const mutated = new Uint8Array(originalBytes);
  const view = new DataView(mutated.buffer, mutated.byteOffset, mutated.byteLength);
  const cmdCount = view.getUint32(8, true);

  const cmdSizes = {
    1: 14, // CREATE_BUFFER
    2: 18, // WRITE_BUFFER
    3: 34, // CREATE_PIPELINE
    4: 46, // RENDER_PASS
    5: 26, // COPY_TEXTURE_TO_BUFFER
    6: 22, // CREATE_TEXTURE
    7: 10, // RECORD_BUNDLE
    8: 14, // EXECUTE_BUNDLES
    9: 26, // SET_VIEWPORT
    10: 18, // SET_SCISSOR_RECT
  };

  let cursor = 16;
  const scissorOffsets = [];
  for (let i = 0; i < cmdCount; i++) {
    const op = view.getUint16(cursor, true);
    if (op === OPCODE_SET_SCISSOR_RECT) {
      scissorOffsets.push(cursor);
    }
    const size = cmdSizes[op];
    if (!size) {
      throw new Error(`mutateRustScissorPacket: unknown opcode ${op} at command ${i}`);
    }
    cursor += size;
  }

  if (scissorOffsets.length <= scissorIndex) {
    throw new Error(
      `Expected at least ${scissorIndex + 1} SET_SCISSOR_RECT commands in Rust packet, found ${scissorOffsets.length}`,
    );
  }

  // Mutate the requested scissor command: set width
  // Command layout: opcode(2), x(4), y(4), width(4), height(4)
  const targetOffset = scissorOffsets[scissorIndex];
  view.setUint32(targetOffset + 10, newWidth, true);

  return mutated;
}

export async function testViewportScissor(host, wasmPacketFn) {
  const device = host.device;
  if (!device) throw new Error("testViewportScissor: device not initialized");

  // 1. Missing Rust packet function is an immediate honest failure
  if (typeof wasmPacketFn !== "function") {
    throw new Error(
      "testViewportScissor: missing required Rust packet function (gpu_bridge_build_nested_viewport_scissor_packet)",
    );
  }

  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const readbackSize = bytesPerRow * height;

  // 2. Decoder checks: orphan SetViewport / SetScissorRect outside active pass must reject
  let rejectedViewport = false;
  try {
    await host.executePacket(buildOrphanViewportPacket(OPCODE_SET_VIEWPORT));
  } catch (err) {
    if (/SetViewport: no active render pass/i.test(err.message)) {
      rejectedViewport = true;
    }
  }
  if (!rejectedViewport) {
    throw new Error(
      "Negative control failed: SetViewport outside active render pass was not rejected",
    );
  }

  let rejectedScissor = false;
  try {
    await host.executePacket(buildOrphanViewportPacket(OPCODE_SET_SCISSOR_RECT));
  } catch (err) {
    if (/SetScissorRect: no active render pass/i.test(err.message)) {
      rejectedScissor = true;
    }
  }
  if (!rejectedScissor) {
    throw new Error(
      "Negative control failed: SetScissorRect outside active render pass was not rejected",
    );
  }

  // 3. Obtain real Rust submission packet
  const rustPacket = wasmPacketFn();
  if (!(rustPacket instanceof Uint8Array) || rustPacket.byteLength === 0) {
    throw new Error(
      "Rust Wasm viewport/scissor packet generator returned empty or non-Uint8Array packet",
    );
  }

  // 4. Render independent direct WebGPU reference (both Target 10 and inner Target 11)
  const direct = await renderDirectViewportScissorReference(device, width, height);

  // 5. Positive execution of Rust packet through bridge
  await host.executePacket(rustPacket);

  // 5a. Verify Target 10 (Buffer 20: outer prefix + resumed pass)
  const bridgePixels10 = await host.readbackBuffer(20, readbackSize);
  if (bridgePixels10.byteLength !== direct.target10Pixels.byteLength) {
    throw new Error(
      `Length mismatch on Target 10: bridge ${bridgePixels10.byteLength} vs direct ${direct.target10Pixels.byteLength}`,
    );
  }

  let diffCount10 = 0;
  let nonZeroCount10 = 0;
  for (let i = 0; i < bridgePixels10.byteLength; i++) {
    if (bridgePixels10[i] !== 0) nonZeroCount10++;
    if (bridgePixels10[i] !== direct.target10Pixels[i]) diffCount10++;
  }
  if (nonZeroCount10 === 0) {
    throw new Error("Rendered image on Target 10 is completely empty/black");
  }
  if (diffCount10 > 0) {
    throw new Error(
      `Pixel mismatch on Target 10 between bridge and direct reference: ${diffCount10} differences`,
    );
  }

  // Spatial sample points on Target 10:
  // - (24, 32): inside left-half scissor [0, 0, 32, 64] and tri1 -> Red
  const offsetLeft = 32 * bytesPerRow + 24 * 4;
  if (bridgePixels10[offsetLeft] < 200 || bridgePixels10[offsetLeft + 2] > 50) {
    throw new Error(
      `Target 10 sample (24, 32) expected Red, got [${bridgePixels10.slice(offsetLeft, offsetLeft + 4)}]`,
    );
  }
  // - (56, 32): inside resumed right-half scissor [32, 0, 32, 64] and tri2 -> Blue
  const offsetRight = 32 * bytesPerRow + 56 * 4;
  if (bridgePixels10[offsetRight + 2] < 200 || bridgePixels10[offsetRight] > 50) {
    throw new Error(
      `Target 10 sample (56, 32) expected Blue, got [${bridgePixels10.slice(offsetRight, offsetRight + 4)}]`,
    );
  }
  // - (2, 2): outside triangles -> Black (clear color)
  const offsetClear = 2 * bytesPerRow + 2 * 4;
  if (
    bridgePixels10[offsetClear] !== 0 ||
    bridgePixels10[offsetClear + 1] !== 0 ||
    bridgePixels10[offsetClear + 2] !== 0
  ) {
    throw new Error(
      `Target 10 sample (2, 2) expected Black clear, got [${bridgePixels10.slice(offsetClear, offsetClear + 4)}]`,
    );
  }

  // 5b. Verify inner Target 11 (Buffer 21: nested inner pass per Mail 7734)
  if (!host.buffers.has(21)) {
    throw new Error(
      "testViewportScissor: packet does not create readback buffer 21 for inner target 11",
    );
  }
  const bridgePixels11 = await host.readbackBuffer(21, readbackSize);
  if (bridgePixels11.byteLength !== direct.target11Pixels.byteLength) {
    throw new Error(
      `Length mismatch on Target 11: bridge ${bridgePixels11.byteLength} vs direct ${direct.target11Pixels.byteLength}`,
    );
  }

  let diffCount11 = 0;
  let nonZeroCount11 = 0;
  for (let i = 0; i < bridgePixels11.byteLength; i++) {
    if (bridgePixels11[i] !== 0) nonZeroCount11++;
    if (bridgePixels11[i] !== direct.target11Pixels[i]) diffCount11++;
  }
  if (nonZeroCount11 === 0) {
    throw new Error(
      "Rendered image on inner Target 11 is completely empty/black (missing green inner rectangle)",
    );
  }
  if (diffCount11 > 0) {
    throw new Error(
      `Pixel mismatch on inner Target 11 between bridge and direct reference: ${diffCount11} differences`,
    );
  }

  // Spatial sample points on Target 11:
  // - (24, 32): inside centered box [16, 16, 32, 32] and tri1 -> Green
  if (
    bridgePixels11[offsetLeft + 1] < 200 ||
    bridgePixels11[offsetLeft] > 50 ||
    bridgePixels11[offsetLeft + 2] > 50
  ) {
    throw new Error(
      `Target 11 sample (24, 32) expected Green, got [${bridgePixels11.slice(offsetLeft, offsetLeft + 4)}]`,
    );
  }
  // - (8, 8): outside centered box -> Black clear
  const offset8_8 = 8 * bytesPerRow + 8 * 4;
  if (
    bridgePixels11[offset8_8] !== 0 ||
    bridgePixels11[offset8_8 + 1] !== 0 ||
    bridgePixels11[offset8_8 + 2] !== 0
  ) {
    throw new Error(
      `Target 11 sample (8, 8) expected Black clear, got [${bridgePixels11.slice(offset8_8, offset8_8 + 4)}]`,
    );
  }
  // - (56, 32): outside centered box -> Black clear
  if (
    bridgePixels11[offsetRight] !== 0 ||
    bridgePixels11[offsetRight + 1] !== 0 ||
    bridgePixels11[offsetRight + 2] !== 0
  ) {
    throw new Error(
      `Target 11 sample (56, 32) expected Black clear, got [${bridgePixels11.slice(offsetRight, offsetRight + 4)}]`,
    );
  }

  // 6. Negative controls: Planted mutations in real Rust packet must fail comparison
  // 6a. Mutate outer resumed scissor in Pass 3 (Target 10) -> must fail Target 10 comparison
  const mutatedOuter = mutateRustScissorPacket(rustPacket, 2, 0);
  const negBridgeOuter = new WebGpuBridgeHost();
  await negBridgeOuter.negotiateAndCreateDevice({ requiredFeatures: [] });
  await negBridgeOuter.executePacket(mutatedOuter);
  const negPixels10 = await negBridgeOuter.readbackBuffer(20, readbackSize);
  let negDiff10 = 0;
  for (let i = 0; i < negPixels10.byteLength; i++) {
    if (negPixels10[i] !== direct.target10Pixels[i]) negDiff10++;
  }
  if (negDiff10 === 0) {
    throw new Error(
      "Negative control failed: mutated outer scissor packet unexpectedly matched direct reference on Target 10",
    );
  }

  // 6b. Mutate inner nested scissor in Pass 2 (Target 11) -> must fail Target 11 comparison
  const mutatedInner = mutateRustScissorPacket(rustPacket, 1, 0);
  const negBridgeInner = new WebGpuBridgeHost();
  await negBridgeInner.negotiateAndCreateDevice({ requiredFeatures: [] });
  await negBridgeInner.executePacket(mutatedInner);
  const negPixels11 = await negBridgeInner.readbackBuffer(21, readbackSize);
  let negDiff11 = 0;
  for (let i = 0; i < negPixels11.byteLength; i++) {
    if (negPixels11[i] !== direct.target11Pixels[i]) negDiff11++;
  }
  if (negDiff11 === 0) {
    throw new Error(
      "Negative control failed: mutated inner scissor packet unexpectedly matched direct reference on Target 11",
    );
  }

  return `Rust nested viewport/scissor packet rendered pixel-identical on outer Target 10 (${bridgePixels10.byteLength}B, verified red-left/blue-right/black-outer) and inner Target 11 (${bridgePixels11.byteLength}B, verified green-center/black-outer); planted outer mutation (${negDiff10} diffs) and inner mutation (${negDiff11} diffs) both detected`;
}
