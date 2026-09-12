/**
 * tests/fixtures/gpu_bridge/depth_test.js
 *
 * Real browser proof for WebGPU depth attachments, depth testing, and multi-pass
 * depth persistence through compiled Rust/Wasm packet -> WebGpuBridgeHost.
 *
 * Verification Requirements (§6.1, §6.7, §8.5):
 * 1. Near/Far overlapping triangles lowered from actual Rust graph/packet builder.
 * 2. Draw-order swap must preserve near color (near z=0.2 green, far z=0.8 red).
 * 3. Multi-pass depth load (LoadOp::Load) must preserve earlier depth across passes.
 * 4. Planted negative (disabled depth / wrong compare) must strictly diverge from
 *    the depth-enabled independent direct-WebGPU reference.
 */

import { WebGpuBridgeHost } from "./bridge_runtime.js";

export function differs(a, b) {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return true;
  }
  return false;
}

/**
 * Genuinely independent direct WebGPU oracle reference for overlapping depth.
 * Issues raw WebGPU API calls directly; does NOT use F3D Wasm or bridge decoders.
 *
 * @param {GPUDevice} device
 * @param {object} options
 * @param {number} [options.scenario=0] - 0: Near then Far, 1: Far then Near, 2: Multi-pass Load, 3: Disabled Depth
 * @param {number} [options.width=64]
 * @param {number} [options.height=64]
 * @returns {Promise<Uint8Array>}
 */
export async function directDepthReference(device, options = {}) {
  const {
    scenario = 0,
    width = 64,
    height = 64,
  } = options;

  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;

  // 1. Target color texture (rgba8unorm)
  const colorTexture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // 2. Depth attachment texture (depth32float)
  const depthTexture = device.createTexture({
    size: [width, height, 1],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // 3. Readback buffer
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Geometry:
  // Far triangle: z = 0.8, Red [1.0, 0.0, 0.0, 1.0]
  // Near triangle: z = 0.2, Green [0.0, 1.0, 0.0, 1.0]
  // Stride 28 bytes: position vec3<f32> (12B) + color vec4<f32> (16B)
  const farVertices = new Float32Array([
    // x,    y,    z,    r,   g,   b,   a
     0.0,  0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
    -0.6, -0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
     0.6, -0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
  ]);

  const nearVertices = new Float32Array([
    // x,    y,    z,    r,   g,   b,   a
     0.0,  0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
    -0.5, -0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
     0.5, -0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
  ]);

  const farBuffer = device.createBuffer({
    size: farVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(farBuffer, 0, farVertices);

  const nearBuffer = device.createBuffer({
    size: nearVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nearBuffer, 0, nearVertices);

  const shaderModule = device.createShaderModule({
    code: `
      struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) color: vec4<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      };

      @vertex
      fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4<f32>(in.position, 1.0);
        out.color = in.color;
        return out;
      }

      @fragment
      fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        return in.color;
      }
    `,
  });

  const isDepthDisabledNegative = scenario === 3;
  const isReadOnlyScenario = scenario === 4;
  const depthCompare = isDepthDisabledNegative ? "always" : "less";
  const depthWriteEnabled = !isDepthDisabledNegative;

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 28,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: "rgba8unorm" }],
    },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled,
      depthCompare,
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
  });

  const pipelineReadOnly = isReadOnlyScenario
    ? device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: shaderModule,
          entryPoint: "vs_main",
          buffers: [
            {
              arrayStride: 28,
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 12, format: "float32x4" },
              ],
            },
          ],
        },
        fragment: {
          module: shaderModule,
          entryPoint: "fs_main",
          targets: [{ format: "rgba8unorm" }],
        },
        depthStencil: {
          format: "depth32float",
          depthWriteEnabled: false,
          depthCompare: "less",
        },
        primitive: {
          topology: "triangle-list",
          cullMode: "none",
        },
      })
    : null;

  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const encoder = device.createCommandEncoder();

    if (scenario === 4) {
      // Scenario 4: Focused Read-Only Depth Counterexample
      // Pass 1: Clear color black, clear depth 1.0, draw Near (Green, z=0.2), store depth
      const pass1 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass1.setPipeline(pipeline);
      pass1.setVertexBuffer(0, nearBuffer);
      pass1.draw(3, 1, 0, 0);
      pass1.end();

      // Pass 2: Load color, depthReadOnly = true (omitted depthLoadOp, depthStoreOp, depthClearValue)
      // Pipeline with depthWriteEnabled = false. Draw Far (Red, z=0.8).
      // Depth test against Near (0.2) rejects Far, preserving Green.
      const pass2DepthAttachment = {
        view: depthTexture.createView(),
        depthReadOnly: true,
      };

      const pass2 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: pass2DepthAttachment,
      });
      pass2.setPipeline(pipelineReadOnly);
      pass2.setVertexBuffer(0, farBuffer);
      pass2.draw(3, 1, 0, 0);
      pass2.end();
    } else if (scenario === 5) {
      // Scenario 5: Empty Depth Clear Pass with Color Load Counterexample
      // Pass 1: Clear color black, clear depth 1.0, draw Near (Green, z=0.2), store depth
      const pass1 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass1.setPipeline(pipeline);
      pass1.setVertexBuffer(0, nearBuffer);
      pass1.draw(3, 1, 0, 0);
      pass1.end();

      // Pass 2: Empty depth clear pass! Load color, Clear depth to 1.0, ZERO draws
      const pass2 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass2.end();

      // Pass 3: Load color, Load depth, draw Far (Red, z=0.8).
      // Because depth was reset to 1.0, Far passes depth test and draws Red over Green!
      const pass3 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      pass3.setPipeline(pipeline);
      pass3.setVertexBuffer(0, farBuffer);
      pass3.draw(3, 1, 0, 0);
      pass3.end();
    } else if (scenario === 2) {
      // Scenario 2: Multi-Pass Depth Persistence
      // Pass 1: Clear color to black, clear depth to 1.0, draw Near (Green, z=0.2), store depth
      const pass1 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass1.setPipeline(pipeline);
      pass1.setVertexBuffer(0, nearBuffer);
      pass1.draw(3, 1, 0, 0);
      pass1.end();

      // Pass 2: Load color, LOAD depth (LoadOp::Load), draw Far (Red, z=0.8)
      // Because depth was preserved, Far (z=0.8) is rejected by depth test against Near (z=0.2)
      const pass2 = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      pass2.setPipeline(pipeline);
      pass2.setVertexBuffer(0, farBuffer);
      pass2.draw(3, 1, 0, 0);
      pass2.end();
    } else {
      // Scenario 0, 1, 3: Single render pass
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorTexture.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(pipeline);

      if (scenario === 0 || scenario === 3) {
        // Near then Far
        pass.setVertexBuffer(0, nearBuffer);
        pass.draw(3, 1, 0, 0);
        pass.setVertexBuffer(0, farBuffer);
        pass.draw(3, 1, 0, 0);
      } else if (scenario === 1) {
        // Far then Near (Draw-order swap)
        pass.setVertexBuffer(0, farBuffer);
        pass.draw(3, 1, 0, 0);
        pass.setVertexBuffer(0, nearBuffer);
        pass.draw(3, 1, 0, 0);
      }
      pass.end();
    }

    encoder.copyTextureToBuffer(
      { texture: colorTexture },
      { buffer: readbackBuffer, bytesPerRow },
      [width, height, 1]
    );

    device.queue.submit([encoder.finish()]);

    const validation = device.popErrorScope();
    scopeOpen = false;
    const error = await validation;
    if (error) {
      throw new Error(`Direct depth reference validation error: ${error.message}`);
    }

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    return pixels;
  } finally {
    if (scopeOpen) await device.popErrorScope();
    colorTexture.destroy();
    depthTexture.destroy();
    readbackBuffer.destroy();
    nearBuffer.destroy();
    farBuffer.destroy();
  }
}

/**
 * Executes the bounded depth pixel regression suite.
 *
 * Exercises Rust graph-lowered depth packet against independent direct WebGPU reference:
 * 1. Checkpoint 1 (Near then Far): Depth test less discards Far, producing Green.
 * 2. Checkpoint 2 (Far then Near): Near overwrites Far, producing Green (draw-order swap preserved).
 * 3. Checkpoint 3 (Multi-Pass Load): Pass 2 loads depth from Pass 1, discards Far, producing Green.
 * 4. Checkpoint 4 (Negative Control): Planted disabled depth produces Red, diverging from reference.
 *
 * @param {WebGpuBridgeHost} bridgeHost
 * @param {object} wasmExports
 * @returns {Promise<string>}
 */
export async function testDepthScene(bridgeHost, wasmExports) {
  const buildDepthFn =
    wasmExports.f3d_build_overlapping_depth_packet ||
    wasmExports.gpu_bridge_build_overlapping_depth_packet;

  if (typeof buildDepthFn !== "function") {
    throw new Error(
      "Missing required Wasm depth export: f3d_build_overlapping_depth_packet / gpu_bridge_build_overlapping_depth_packet"
    );
  }

  const device = bridgeHost.device;
  const width = 64;
  const height = 64;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const centerIdx = 32 * bytesPerRow + 32 * 4;

  // ---------------------------------------------------------------------------
  // Checkpoint 1: Scenario 0 (Near then Far) vs Independent Direct Reference
  // ---------------------------------------------------------------------------
  const packet0 = buildDepthFn(0);
  await bridgeHost.executePacket(packet0);
  const candidatePixels0 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refPixels0 = await directDepthReference(device, { scenario: 0, width, height });

  if (differs(candidatePixels0, refPixels0)) {
    throw new Error("Checkpoint 1 failed: candidate Near-then-Far depth pixels differ from independent direct WebGPU reference");
  }

  const center0 = [
    candidatePixels0[centerIdx],
    candidatePixels0[centerIdx + 1],
    candidatePixels0[centerIdx + 2],
    candidatePixels0[centerIdx + 3],
  ];
  if (center0[0] > 50 || center0[1] < 200 || center0[2] > 50 || center0[3] !== 255) {
    throw new Error(`Checkpoint 1 failed: expected center Green [0, 255, 0, 255], got [${center0}]`);
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 2: Scenario 1 (Far then Near - Draw-Order Swap Preservation)
  // ---------------------------------------------------------------------------
  const packet1 = buildDepthFn(1);
  await bridgeHost.executePacket(packet1);
  const candidatePixels1 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refPixels1 = await directDepthReference(device, { scenario: 1, width, height });

  if (differs(candidatePixels1, refPixels1)) {
    throw new Error("Checkpoint 2 failed: candidate Far-then-Near depth pixels differ from independent direct WebGPU reference");
  }

  // Draw-order swap must preserve near color: Checkpoint 1 and 2 must match bit-for-bit
  if (differs(candidatePixels1, candidatePixels0)) {
    throw new Error("Checkpoint 2 failed: draw-order swap produced different pixels; near geometry was not preserved across draw order");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 3: Scenario 2 (Multi-Pass Depth Persistence via LoadOp::Load)
  // ---------------------------------------------------------------------------
  const packet2 = buildDepthFn(2);
  await bridgeHost.executePacket(packet2);
  const candidatePixels2 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refPixels2 = await directDepthReference(device, { scenario: 2, width, height });

  if (differs(candidatePixels2, refPixels2)) {
    throw new Error("Checkpoint 3 failed: multi-pass depth load candidate pixels differ from independent direct WebGPU reference");
  }

  // Multi-pass depth load must preserve earlier depth: candidatePixels2 must match candidatePixels0
  if (differs(candidatePixels2, candidatePixels0)) {
    throw new Error("Checkpoint 3 failed: depth load across second pass failed to preserve earlier depth; far draw was not discarded");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 4: Scenario 3 (Negative Control - Planted Disabled Depth Mismatch)
  // ---------------------------------------------------------------------------
  const packet3 = buildDepthFn(3);
  await bridgeHost.executePacket(packet3);
  const candidatePixels3 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);

  // In scenario 3, depth testing is disabled, so Far (Red) overwrites Near (Green)
  const center3 = [
    candidatePixels3[centerIdx],
    candidatePixels3[centerIdx + 1],
    candidatePixels3[centerIdx + 2],
    candidatePixels3[centerIdx + 3],
  ];
  if (center3[0] < 200 || center3[1] > 50 || center3[2] > 50) {
    throw new Error(`Checkpoint 4 failed: disabled-depth candidate expected Red [255, 0, 0, 255] at center, got [${center3}]`);
  }

  // Planted negative must strictly diverge from depth-enabled reference
  if (!differs(candidatePixels3, refPixels0)) {
    throw new Error("Checkpoint 4 failed: planted disabled-depth packet falsely matched depth-enabled reference");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 5: Scenario 4 (Focused Read-Only Depth Counterexample)
  // ---------------------------------------------------------------------------
  const packet4 = buildDepthFn(4);
  await bridgeHost.executePacket(packet4);
  const candidatePixels4 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refPixels4 = await directDepthReference(device, { scenario: 4, width, height });

  if (differs(candidatePixels4, refPixels4)) {
    throw new Error("Checkpoint 5 failed: candidate read-only depth pixels differ from independent direct WebGPU reference");
  }

  const center4 = [
    candidatePixels4[centerIdx],
    candidatePixels4[centerIdx + 1],
    candidatePixels4[centerIdx + 2],
    candidatePixels4[centerIdx + 3],
  ];
  if (center4[0] > 50 || center4[1] < 200 || center4[2] > 50 || center4[3] !== 255) {
    const isRed = center4[0] >= 200 && center4[1] <= 50;
    const diagnosis = isRed
      ? "Far geometry (Red) rendered; prior depth (Green, 0.2) was NOT loaded into read-only pass (behaved as Clear/DontCare instead of Load)"
      : `unexpected center pixel [${center4}]`;
    throw new Error(`Checkpoint 5 failed: expected center Green [0, 255, 0, 255] in read-only depth pass, got [${center4}]. Diagnosis: ${diagnosis}`);
  }

  if (differs(candidatePixels4, candidatePixels0)) {
    throw new Error("Checkpoint 5 failed: read-only depth pass modified depth buffer or failed to preserve near geometry");
  }

  // ---------------------------------------------------------------------------
  // Checkpoint 6: Scenario 5 (Empty Depth Clear Pass with Color Load Counterexample)
  // ---------------------------------------------------------------------------
  const packet5 = buildDepthFn(5);
  await bridgeHost.executePacket(packet5);
  const candidatePixels5 = await bridgeHost.readbackBuffer(20, bytesPerRow * height);
  const refPixels5 = await directDepthReference(device, { scenario: 5, width, height });

  if (differs(candidatePixels5, refPixels5)) {
    throw new Error("Checkpoint 6 failed: candidate empty depth clear pass pixels differ from independent direct WebGPU reference");
  }

  const center5 = [
    candidatePixels5[centerIdx],
    candidatePixels5[centerIdx + 1],
    candidatePixels5[centerIdx + 2],
    candidatePixels5[centerIdx + 3],
  ];
  if (center5[0] < 200 || center5[1] > 50 || center5[2] > 50) {
    throw new Error(`Checkpoint 6 failed: expected center Red [255, 0, 0, 255] after empty depth clear pass, got [${center5}]`);
  }

  // Must strictly diverge from refPixels0 (which is Green)
  if (!differs(candidatePixels5, refPixels0)) {
    throw new Error("Checkpoint 6 failed: empty depth clear pass falsely matched un-cleared depth reference; depth was not cleared without draws");
  }

  return "WebGPU depth regression verified: near/far overlapping triangles lowered from actual Rust graph packet; draw-order swap preserves near color; multi-pass depth load preserves earlier depth; disabled-depth negative strictly diverges; readonly depth descriptor verified; empty depth clear pass verified";
}
