/**
 * variant_preconditions.js - Per-Variant Correctness Preconditions
 * (§8.5, §16.7, §21, bead f3d-03-reference-profiles-and-bridge-tmt.5)
 *
 * Enforces two mandatory correctness counterexample preconditions for EVERY bridge variant:
 * 1. RED-A / BLUE-B Snapshot Isolation in a Single Queue Submission:
 *    - Positive check: per-use versioned writes to Target A (Red) and Target B (Blue) in one submission.
 *      Target A center pixel must be Red; Target B center pixel must be Blue.
 *    - Negative control: single-slot unversioned reuse (overwriting before submit) must FAIL isolation,
 *      proving queue hazard detection.
 *
 * 2. BUNDLE-THEN-DIRECT State Reset Invalidation:
 *    - Records and executes a render bundle (Triangle 1, Green at offset 0).
 *    - Direct draw in the same render pass (Triangle 2, Blue at offset 256).
 *    - executeBundles clears pass state per WebGPU specification; the direct draw MUST re-bind
 *      pipeline, vertex buffer, and dynamic uniform offset inside the same pass.
 *    - Observable pixels: Left triangle is Green, Right triangle is Blue, Background is Black.
 *
 * Variants tested:
 * - "direct": Direct JS WebGPU
 * - "bulk": Rust/Wasm packet encoder; JS WebGPU decoder
 * - "chatty": Rust/Wasm callback loop; JS WebGPU submission
 * - "generated": Actual static generator output for draw bindings; JS resource/pass setup, no Wasm data packing
 *
 * Correctness only; no timing. Lists any excluded variants explicitly.
 */

const WIDTH = 64;
const HEIGHT = 64;
const BYTES_PER_ROW = 256;
const READBACK_SIZE = BYTES_PER_ROW * HEIGHT; // 16384 bytes
const CENTER_OFFSET = 32 * BYTES_PER_ROW + 32 * 4; // 8320
const LEFT_GREEN_OFFSET = 48 * BYTES_PER_ROW + 19 * 4; // 12364 (col 19, row 48)
const RIGHT_BLUE_OFFSET = 48 * BYTES_PER_ROW + 51 * 4; // 12492 (col 51, row 48)
const BACKGROUND_OFFSET = 10 * BYTES_PER_ROW + 10 * 4; // 2600 (col 10, row 10)

const RED_BLUE_SHADER_CODE = `
struct ColorUniform {
    color: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> u: ColorUniform;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
`;

const BUNDLE_DIRECT_SHADER_CODE = `
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
`;

const TRI1_VERTICES = new Float32Array([
  // x,    y,    z,   u,   v
  -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0,
]);

const TRI2_VERTICES = new Float32Array([
  // x,    y,    z,   u,   v
  0.0, -1.0, 0.0, 0.5, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
]);

function buildColorUniformData(c0, c1 = null) {
  const buf = new Float32Array(128); // 512 bytes = 2 slots of 256 bytes
  buf[0] = c0[0];
  buf[1] = c0[1];
  buf[2] = c0[2];
  buf[3] = c0[3];
  if (c1) {
    buf[64] = c1[0];
    buf[65] = c1[1];
    buf[66] = c1[2];
    buf[67] = c1[3];
  }
  return buf;
}

async function readbackStagingBuffer(device, readbackBuffer) {
  await readbackBuffer.mapAsync(GPUMapMode.READ, 0, READBACK_SIZE);
  const mapped = readbackBuffer.getMappedRange(0, READBACK_SIZE);
  const result = new Uint8Array(mapped.slice(0));
  readbackBuffer.unmap();
  return result;
}

// Built via RCH using generateStaticSubmissionSource(1, 256, firstDraw).
// Load only for this variant so missing artifacts cannot break unrelated lanes.
let generatedDraws;
async function loadGeneratedDraws() {
  if (generatedDraws) return;
  const modules = await Promise.all([
    import("/out/browser-probe/static_submission_precondition0.js"),
    import("/out/browser-probe/static_submission_precondition1.js"),
  ]);
  const draws = modules.map((module) => module.executeStaticSubmission1);
  if (draws.some((draw) => typeof draw !== "function")) {
    throw new Error("Missing executeStaticSubmission1 in generated precondition modules");
  }
  generatedDraws = draws;
}

function executeGeneratedDraw(pass, bindGroup, dynamicOffset) {
  if (dynamicOffset !== 0 && dynamicOffset !== 256) {
    throw new RangeError(`No generated precondition draw for offset ${dynamicOffset}`);
  }
  generatedDraws[dynamicOffset / 256](pass, bindGroup);
}

function executeGeneratedRedBluePass(pass, pipeline, bindGroup, dynamicOffset) {
  pass.setPipeline(pipeline);
  executeGeneratedDraw(pass, bindGroup, dynamicOffset);
}

function executeGeneratedBundleRecord(
  bundleEncoder,
  pipeline,
  vertexBuffer,
  bindGroup,
  dynamicOffset,
) {
  bundleEncoder.setPipeline(pipeline);
  bundleEncoder.setVertexBuffer(0, vertexBuffer);
  executeGeneratedDraw(bundleEncoder, bindGroup, dynamicOffset);
}

function executeGeneratedBundleDirectPass(pass, pipeline, vertexBuffer, bindGroup, dynamicOffset) {
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  executeGeneratedDraw(pass, bindGroup, dynamicOffset);
}

/**
 * 1. RED-A / BLUE-B Check Implementation
 */
async function runRedABlueBForVariant(variantKey, bridge, wasmExports) {
  const device = bridge.device;
  if (variantKey === "generated") await loadGeneratedDraws();

  if (variantKey === "bulk") {
    // Bulk packet implementation using canonical Wasm exports
    const buildRedBlueFn =
      wasmExports.f3d_build_red_a_blue_b_packet || wasmExports.gpu_bridge_build_red_blue_packet;
    if (typeof buildRedBlueFn !== "function")
      throw new Error("Missing f3d_build_red_a_blue_b_packet export");

    // Positive check (versioned = true)
    let colorA, colorB;
    try {
      const positivePacket = buildRedBlueFn(true);
      await bridge.executePacket(positivePacket);
      const pixelsA = await bridge.readbackBuffer(40, READBACK_SIZE);
      const pixelsB = await bridge.readbackBuffer(41, READBACK_SIZE);
      colorA = [
        pixelsA[CENTER_OFFSET],
        pixelsA[CENTER_OFFSET + 1],
        pixelsA[CENTER_OFFSET + 2],
        pixelsA[CENTER_OFFSET + 3],
      ];
      colorB = [
        pixelsB[CENTER_OFFSET],
        pixelsB[CENTER_OFFSET + 1],
        pixelsB[CENTER_OFFSET + 2],
        pixelsB[CENTER_OFFSET + 3],
      ];
    } finally {
      for (const id of [1, 40, 41]) {
        bridge.buffers.get(id)?.destroy();
        bridge.buffers.delete(id);
        bridge.bufferEpochs?.delete(id);
      }
      for (const id of [30, 31]) {
        bridge.textures.get(id)?.destroy();
        bridge.textures.delete(id);
      }
      bridge.pipelines.delete(200);
    }

    const positivePassed = colorA[0] > 200 && colorA[2] < 50 && colorB[2] > 200 && colorB[0] < 50;
    if (!positivePassed) {
      throw new Error(
        `Bulk positive isolation failed: Target A was [${colorA}], Target B was [${colorB}]`,
      );
    }

    // Negative control (versioned = false: unversioned single-slot reuse)
    let hazardColorA, hazardColorB;
    try {
      const negativePacket = buildRedBlueFn(false);
      await bridge.executePacket(negativePacket);
      const hazardPixelsA = await bridge.readbackBuffer(40, READBACK_SIZE);
      const hazardPixelsB = await bridge.readbackBuffer(41, READBACK_SIZE);
      hazardColorA = [
        hazardPixelsA[CENTER_OFFSET],
        hazardPixelsA[CENTER_OFFSET + 1],
        hazardPixelsA[CENTER_OFFSET + 2],
        hazardPixelsA[CENTER_OFFSET + 3],
      ];
      hazardColorB = [
        hazardPixelsB[CENTER_OFFSET],
        hazardPixelsB[CENTER_OFFSET + 1],
        hazardPixelsB[CENTER_OFFSET + 2],
        hazardPixelsB[CENTER_OFFSET + 3],
      ];
    } finally {
      for (const id of [1, 40, 41]) {
        bridge.buffers.get(id)?.destroy();
        bridge.buffers.delete(id);
        bridge.bufferEpochs?.delete(id);
      }
      for (const id of [30, 31]) {
        bridge.textures.get(id)?.destroy();
        bridge.textures.delete(id);
      }
      bridge.pipelines.delete(200);
    }

    // Require both hazard targets to be opaque (> 200)
    if (hazardColorA[3] <= 200 || hazardColorB[3] <= 200) {
      throw new Error(
        `Bulk negative control targets must be opaque (alpha > 200): Target A alpha=${hazardColorA[3]}, Target B alpha=${hazardColorB[3]}`,
      );
    }

    // Require hazard A and hazard B to be the SAME color: both Red for bulk
    const bothRed =
      hazardColorA[0] > 200 &&
      hazardColorA[2] < 50 &&
      hazardColorB[0] > 200 &&
      hazardColorB[2] < 50;
    const sameColor =
      Math.abs(hazardColorA[0] - hazardColorB[0]) < 10 &&
      Math.abs(hazardColorA[1] - hazardColorB[1]) < 10 &&
      Math.abs(hazardColorA[2] - hazardColorB[2]) < 10 &&
      Math.abs(hazardColorA[3] - hazardColorB[3]) < 10;
    if (!bothRed || !sameColor) {
      throw new Error(
        `Bulk negative control failed: expected both targets to be Red (shared-slot offset), got Target A=[${hazardColorA}], Target B=[${hazardColorB}]`,
      );
    }

    const isolationFailed = !(
      hazardColorA[0] > 200 &&
      hazardColorA[2] < 50 &&
      hazardColorB[2] > 200 &&
      hazardColorB[0] < 50
    );
    if (!isolationFailed) {
      throw new Error(
        `Bulk negative control failed: unversioned packet falsely passed isolation (A=[${hazardColorA}], B=[${hazardColorB}])`,
      );
    }

    return {
      status: "PASS",
      color_a: colorA,
      color_b: colorB,
      negative_control: {
        status: "PASS",
        hazard_color_a: hazardColorA,
        hazard_color_b: hazardColorB,
        expected_hazard_color: "red",
        isolation_failed: true,
      },
    };
  }

  // Non-bulk variants: direct, chatty, generated
  const shaderModule = device.createShaderModule({ code: RED_BLUE_SHADER_CODE });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
      },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });

  const targetA = device.createTexture({
    size: [WIDTH, HEIGHT, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const targetB = device.createTexture({
    size: [WIDTH, HEIGHT, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readbackA = device.createBuffer({
    size: READBACK_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const readbackB = device.createBuffer({
    size: READBACK_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  let colorA, colorB;
  let hazardColorA, hazardColorB;
  let validationError = null;
  try {
    // 1. Positive check: 2-slot versioned buffer (Red at 0, Blue at 256)
    const versionedBuffer = device.createBuffer({
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
      const versionedData = buildColorUniformData([1, 0, 0, 1], [0, 0, 1, 1]);
      device.queue.writeBuffer(versionedBuffer, 0, versionedData);

      const versionedBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: versionedBuffer, offset: 0, size: 16 } }],
      });

      const encoder = device.createCommandEncoder();

      const passA = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetA.createView(),
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      if (variantKey === "chatty") {
        const hostObj =
          typeof window !== "undefined"
            ? (window.f3dHost = window.f3dHost || {})
            : (globalThis.f3dHost = globalThis.f3dHost || {});
        const prev = hostObj.drawCall;
        try {
          hostObj.drawCall = () => {
            passA.setPipeline(pipeline);
            passA.setBindGroup(0, versionedBindGroup, [0]);
            passA.draw(3, 1, 0, 0);
          };
          wasmExports.f3d_bridge_chatty_draw_loop(1);
        } finally {
          if (prev !== undefined) hostObj.drawCall = prev;
          else delete hostObj.drawCall;
        }
      } else if (variantKey === "generated") {
        executeGeneratedRedBluePass(passA, pipeline, versionedBindGroup, 0);
      } else {
        passA.setPipeline(pipeline);
        passA.setBindGroup(0, versionedBindGroup, [0]);
        passA.draw(3, 1, 0, 0);
      }
      passA.end();

      const passB = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetB.createView(),
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      if (variantKey === "chatty") {
        const hostObj =
          typeof window !== "undefined"
            ? (window.f3dHost = window.f3dHost || {})
            : (globalThis.f3dHost = globalThis.f3dHost || {});
        const prev = hostObj.drawCall;
        try {
          hostObj.drawCall = () => {
            passB.setPipeline(pipeline);
            passB.setBindGroup(0, versionedBindGroup, [256]);
            passB.draw(3, 1, 0, 0);
          };
          wasmExports.f3d_bridge_chatty_draw_loop(1);
        } finally {
          if (prev !== undefined) hostObj.drawCall = prev;
          else delete hostObj.drawCall;
        }
      } else if (variantKey === "generated") {
        executeGeneratedRedBluePass(passB, pipeline, versionedBindGroup, 256);
      } else {
        passB.setPipeline(pipeline);
        passB.setBindGroup(0, versionedBindGroup, [256]);
        passB.draw(3, 1, 0, 0);
      }
      passB.end();

      encoder.copyTextureToBuffer(
        { texture: targetA },
        { buffer: readbackA, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT, 1],
      );
      encoder.copyTextureToBuffer(
        { texture: targetB },
        { buffer: readbackB, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT, 1],
      );

      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      const pixelsA = await readbackStagingBuffer(device, readbackA);
      const pixelsB = await readbackStagingBuffer(device, readbackB);
      colorA = [
        pixelsA[CENTER_OFFSET],
        pixelsA[CENTER_OFFSET + 1],
        pixelsA[CENTER_OFFSET + 2],
        pixelsA[CENTER_OFFSET + 3],
      ];
      colorB = [
        pixelsB[CENTER_OFFSET],
        pixelsB[CENTER_OFFSET + 1],
        pixelsB[CENTER_OFFSET + 2],
        pixelsB[CENTER_OFFSET + 3],
      ];
    } finally {
      versionedBuffer.destroy();
    }

    const positivePassed = colorA[0] > 200 && colorA[2] < 50 && colorB[2] > 200 && colorB[0] < 50;
    if (!positivePassed) {
      throw new Error(
        `${variantKey} positive isolation failed: Target A was [${colorA}], Target B was [${colorB}]`,
      );
    }

    // 2. Negative control: single-slot unversioned reuse (writes Blue before submission)
    device.pushErrorScope("validation");
    let unversionedBuffer = null;
    try {
      unversionedBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const redData = buildColorUniformData([1, 0, 0, 1]);
      const blueData = buildColorUniformData([0, 0, 1, 1]);

      // Initial write of Red
      device.queue.writeBuffer(unversionedBuffer, 0, redData.subarray(0, 64));

      const unversionedBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: unversionedBuffer, offset: 0, size: 16 } }],
      });

      const encoder = device.createCommandEncoder();

      const passA = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetA.createView(),
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      if (variantKey === "chatty") {
        const hostObj =
          typeof window !== "undefined"
            ? (window.f3dHost = window.f3dHost || {})
            : (globalThis.f3dHost = globalThis.f3dHost || {});
        const prev = hostObj.drawCall;
        try {
          hostObj.drawCall = () => {
            passA.setPipeline(pipeline);
            passA.setBindGroup(0, unversionedBindGroup, [0]);
            passA.draw(3, 1, 0, 0);
          };
          wasmExports.f3d_bridge_chatty_draw_loop(1);
        } finally {
          if (prev !== undefined) hostObj.drawCall = prev;
          else delete hostObj.drawCall;
        }
      } else if (variantKey === "generated") {
        executeGeneratedRedBluePass(passA, pipeline, unversionedBindGroup, 0);
      } else {
        passA.setPipeline(pipeline);
        passA.setBindGroup(0, unversionedBindGroup, [0]);
        passA.draw(3, 1, 0, 0);
      }
      passA.end();

      const passB = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetB.createView(),
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      if (variantKey === "chatty") {
        const hostObj =
          typeof window !== "undefined"
            ? (window.f3dHost = window.f3dHost || {})
            : (globalThis.f3dHost = globalThis.f3dHost || {});
        const prev = hostObj.drawCall;
        try {
          hostObj.drawCall = () => {
            passB.setPipeline(pipeline);
            passB.setBindGroup(0, unversionedBindGroup, [0]);
            passB.draw(3, 1, 0, 0);
          };
          wasmExports.f3d_bridge_chatty_draw_loop(1);
        } finally {
          if (prev !== undefined) hostObj.drawCall = prev;
          else delete hostObj.drawCall;
        }
      } else if (variantKey === "generated") {
        executeGeneratedRedBluePass(passB, pipeline, unversionedBindGroup, 0);
      } else {
        passB.setPipeline(pipeline);
        passB.setBindGroup(0, unversionedBindGroup, [0]);
        passB.draw(3, 1, 0, 0);
      }
      passB.end();

      encoder.copyTextureToBuffer(
        { texture: targetA },
        { buffer: readbackA, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT, 1],
      );
      encoder.copyTextureToBuffer(
        { texture: targetB },
        { buffer: readbackB, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
        [WIDTH, HEIGHT, 1],
      );

      // Hazard: application writes Blue to the unversioned slot BEFORE submission
      device.queue.writeBuffer(unversionedBuffer, 0, blueData.subarray(0, 64));

      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      validationError = await device.popErrorScope();

      const hazardPixelsA = await readbackStagingBuffer(device, readbackA);
      const hazardPixelsB = await readbackStagingBuffer(device, readbackB);
      hazardColorA = [
        hazardPixelsA[CENTER_OFFSET],
        hazardPixelsA[CENTER_OFFSET + 1],
        hazardPixelsA[CENTER_OFFSET + 2],
        hazardPixelsA[CENTER_OFFSET + 3],
      ];
      hazardColorB = [
        hazardPixelsB[CENTER_OFFSET],
        hazardPixelsB[CENTER_OFFSET + 1],
        hazardPixelsB[CENTER_OFFSET + 2],
        hazardPixelsB[CENTER_OFFSET + 3],
      ];
    } catch (err) {
      if (validationError === null) {
        try {
          validationError = await device.popErrorScope();
        } catch (_) {}
      }
      throw err;
    } finally {
      if (unversionedBuffer) {
        unversionedBuffer.destroy();
      }
    }
  } finally {
    targetA.destroy();
    targetB.destroy();
    readbackA.destroy();
    readbackB.destroy();
  }

  if (validationError) {
    throw new Error(
      `${variantKey} negative control validation error: ${validationError.message || validationError}`,
    );
  }

  // Require both hazard targets to be opaque (> 200)
  if (hazardColorA[3] <= 200 || hazardColorB[3] <= 200) {
    throw new Error(
      `${variantKey} negative control targets must be opaque (alpha > 200): Target A alpha=${hazardColorA[3]}, Target B alpha=${hazardColorB[3]}`,
    );
  }

  // Require hazard A and hazard B to be the SAME color: both Blue for non-bulk
  const bothBlue =
    hazardColorA[2] > 200 && hazardColorA[0] < 50 && hazardColorB[2] > 200 && hazardColorB[0] < 50;
  const sameColor =
    Math.abs(hazardColorA[0] - hazardColorB[0]) < 10 &&
    Math.abs(hazardColorA[1] - hazardColorB[1]) < 10 &&
    Math.abs(hazardColorA[2] - hazardColorB[2]) < 10 &&
    Math.abs(hazardColorA[3] - hazardColorB[3]) < 10;
  if (!bothBlue || !sameColor) {
    throw new Error(
      `${variantKey} negative control failed: expected both targets to be Blue (shared-slot overwrite), got Target A=[${hazardColorA}], Target B=[${hazardColorB}]`,
    );
  }

  const isolationFailed = !(
    hazardColorA[0] > 200 &&
    hazardColorA[2] < 50 &&
    hazardColorB[2] > 200 &&
    hazardColorB[0] < 50
  );
  if (!isolationFailed) {
    throw new Error(
      `${variantKey} negative control failed: unversioned write falsely passed isolation (A=[${hazardColorA}], B=[${hazardColorB}])`,
    );
  }

  return {
    status: "PASS",
    color_a: colorA,
    color_b: colorB,
    negative_control: {
      status: "PASS",
      hazard_color_a: hazardColorA,
      hazard_color_b: hazardColorB,
      expected_hazard_color: "blue",
      isolation_failed: true,
    },
  };
}

/**
 * 2. BUNDLE-THEN-DIRECT State Reset Invalidation Implementation
 */
async function runBundleThenDirectForVariant(variantKey, bridge, wasmExports) {
  const device = bridge.device;
  if (variantKey === "generated") await loadGeneratedDraws();

  if (variantKey === "bulk") {
    const buildBundleDirectFn =
      wasmExports.f3d_build_bundle_direct_draw_packet ||
      wasmExports.gpu_bridge_build_bundle_direct_draw_packet;
    if (typeof buildBundleDirectFn !== "function")
      throw new Error("Missing f3d_build_bundle_direct_draw_packet export");

    let greenPixel, bluePixel, bgPixel;
    try {
      const packet = buildBundleDirectFn();
      await bridge.executePacket(packet);
      const pixels = await bridge.readbackBuffer(20, READBACK_SIZE);
      greenPixel = [
        pixels[LEFT_GREEN_OFFSET],
        pixels[LEFT_GREEN_OFFSET + 1],
        pixels[LEFT_GREEN_OFFSET + 2],
        pixels[LEFT_GREEN_OFFSET + 3],
      ];
      bluePixel = [
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
      bridge.bundles.delete(1);
      for (const id of [1, 2, 3, 20]) {
        bridge.buffers.get(id)?.destroy();
        bridge.buffers.delete(id);
        bridge.bufferEpochs?.delete(id);
      }
      bridge.textures.get(10)?.destroy();
      bridge.textures.delete(10);
      bridge.pipelines.delete(200);
    }

    const greenValid = greenPixel[1] > 200 && greenPixel[0] < 50 && greenPixel[2] < 50;
    const blueValid = bluePixel[2] > 200 && bluePixel[0] < 50 && bluePixel[1] < 50;
    const bgValid = bgPixel[0] < 50 && bgPixel[1] < 50 && bgPixel[2] < 50;

    if (!greenValid || !blueValid || !bgValid) {
      throw new Error(
        `Bulk bundle-then-direct check failed: Green=[${greenPixel}], Blue=[${bluePixel}], BG=[${bgPixel}]`,
      );
    }

    return {
      status: "PASS",
      green_pixel: greenPixel,
      blue_pixel: bluePixel,
      background_pixel: bgPixel,
    };
  }

  // Non-bulk variants: direct, chatty, generated
  const shaderModule = device.createShaderModule({ code: BUNDLE_DIRECT_SHADER_CODE });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
      },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
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
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });

  const uniformBuffer = device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformData = buildColorUniformData([0, 1, 0, 1], [0, 0, 1, 1]); // 0=Green, 256=Blue
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const vbBundle = device.createBuffer({
    size: TRI1_VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vbBundle, 0, TRI1_VERTICES);

  const vbDirect = device.createBuffer({
    size: TRI2_VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vbDirect, 0, TRI2_VERTICES);

  const targetTexture = device.createTexture({
    size: [WIDTH, HEIGHT, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: READBACK_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: 0, size: 16 } }],
  });

  let greenPixel, bluePixel, bgPixel;
  try {
    // 1. Pre-record render bundle: draws Triangle 1 with Green (dynamic offset 0)
    const bundleEncoder = device.createRenderBundleEncoder({ colorFormats: ["rgba8unorm"] });
    if (variantKey === "chatty") {
      const hostObj =
        typeof window !== "undefined"
          ? (window.f3dHost = window.f3dHost || {})
          : (globalThis.f3dHost = globalThis.f3dHost || {});
      const prev = hostObj.drawCall;
      try {
        hostObj.drawCall = () => {
          bundleEncoder.setPipeline(pipeline);
          bundleEncoder.setVertexBuffer(0, vbBundle);
          bundleEncoder.setBindGroup(0, bindGroup, [0]);
          bundleEncoder.draw(3, 1, 0, 0);
        };
        wasmExports.f3d_bridge_chatty_draw_loop(1);
      } finally {
        if (prev !== undefined) hostObj.drawCall = prev;
        else delete hostObj.drawCall;
      }
    } else if (variantKey === "generated") {
      executeGeneratedBundleRecord(bundleEncoder, pipeline, vbBundle, bindGroup, 0);
    } else {
      bundleEncoder.setPipeline(pipeline);
      bundleEncoder.setVertexBuffer(0, vbBundle);
      bundleEncoder.setBindGroup(0, bindGroup, [0]);
      bundleEncoder.draw(3, 1, 0, 0);
    }
    const bundle = bundleEncoder.finish();

    // 2. Render pass: execute bundle, clear state via executeBundles, then direct draw
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetTexture.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    // Execute bundle
    pass.executeBundles([bundle]);

    // Empty bundle sequence (spec invariant: also resets pass state)
    pass.executeBundles([]);

    // Direct draw in the same pass: MUST re-bind pipeline, vertex buffer, and bind group
    if (variantKey === "chatty") {
      const hostObj =
        typeof window !== "undefined"
          ? (window.f3dHost = window.f3dHost || {})
          : (globalThis.f3dHost = globalThis.f3dHost || {});
      const prev = hostObj.drawCall;
      try {
        hostObj.drawCall = () => {
          pass.setPipeline(pipeline);
          pass.setVertexBuffer(0, vbDirect);
          pass.setBindGroup(0, bindGroup, [256]);
          pass.draw(3, 1, 0, 0);
        };
        wasmExports.f3d_bridge_chatty_draw_loop(1);
      } finally {
        if (prev !== undefined) hostObj.drawCall = prev;
        else delete hostObj.drawCall;
      }
    } else if (variantKey === "generated") {
      executeGeneratedBundleDirectPass(pass, pipeline, vbDirect, bindGroup, 256);
    } else {
      pass.setPipeline(pipeline);
      pass.setVertexBuffer(0, vbDirect);
      pass.setBindGroup(0, bindGroup, [256]);
      pass.draw(3, 1, 0, 0);
    }
    pass.end();

    encoder.copyTextureToBuffer(
      { texture: targetTexture },
      { buffer: readbackBuffer, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
      [WIDTH, HEIGHT, 1],
    );

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const pixels = await readbackStagingBuffer(device, readbackBuffer);
    greenPixel = [
      pixels[LEFT_GREEN_OFFSET],
      pixels[LEFT_GREEN_OFFSET + 1],
      pixels[LEFT_GREEN_OFFSET + 2],
      pixels[LEFT_GREEN_OFFSET + 3],
    ];
    bluePixel = [
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
    uniformBuffer.destroy();
    vbBundle.destroy();
    vbDirect.destroy();
    targetTexture.destroy();
    readbackBuffer.destroy();
  }

  const greenValid = greenPixel[1] > 200 && greenPixel[0] < 50 && greenPixel[2] < 50;
  const blueValid = bluePixel[2] > 200 && bluePixel[0] < 50 && bluePixel[1] < 50;
  const bgValid = bgPixel[0] < 50 && bgPixel[1] < 50 && bgPixel[2] < 50;

  if (!greenValid || !blueValid || !bgValid) {
    throw new Error(
      `${variantKey} bundle-then-direct check failed: Green=[${greenPixel}], Blue=[${bluePixel}], BG=[${bgPixel}]`,
    );
  }

  return {
    status: "PASS",
    green_pixel: greenPixel,
    blue_pixel: bluePixel,
    background_pixel: bgPixel,
  };
}

/**
 * Main entry point for lane=variant_preconditions.
 */
export async function testVariantPreconditions(bridge, wasmExports) {
  if (!bridge || !bridge.device) throw new Error("WebGpuBridgeHost device not initialized");

  const variants = [
    { key: "direct", owner: "Direct JS WebGPU" },
    { key: "bulk", owner: "Rust/Wasm packet encoder; JS WebGPU decoder" },
    { key: "chatty", owner: "Rust/Wasm callback loop; JS WebGPU submission" },
    {
      key: "generated",
      owner: "Generated static draw bindings; JavaScript resource/pass setup, no Wasm data packing",
    },
  ];

  const results = {
    variants: {},
    excluded_variants: [],
    generated_precondition_scope: "static_submission_generator_output",
    tests: {},
  };

  for (const { key, owner } of variants) {
    const variantRecord = {
      implementation_owner: owner,
      preconditions_met: false,
    };

    // 1. RED-A / BLUE-B Check
    const redBlueTestKey = `precondition_red_a_blue_b_${key}`;
    try {
      const rbRes = await runRedABlueBForVariant(key, bridge, wasmExports);
      variantRecord.red_a_blue_b = rbRes;
      results.tests[redBlueTestKey] = {
        status: "PASS",
        detail: `${key}: Target A is Red (${rbRes.color_a.join(",")}), Target B is Blue (${rbRes.color_b.join(",")}); negative control proved unversioned hazard (expected ${rbRes.negative_control.expected_hazard_color}, Target A=[${rbRes.negative_control.hazard_color_a}], Target B=[${rbRes.negative_control.hazard_color_b}])`,
        implementation_owner: owner,
      };
    } catch (err) {
      variantRecord.red_a_blue_b = { status: "FAIL", error: err.message || String(err) };
      results.tests[redBlueTestKey] = {
        status: "FAIL",
        error: err.message || String(err),
        implementation_owner: owner,
      };
    }

    // 2. BUNDLE-THEN-DIRECT Check
    const bundleDirectTestKey = `precondition_bundle_then_direct_${key}`;
    try {
      const bdRes = await runBundleThenDirectForVariant(key, bridge, wasmExports);
      variantRecord.bundle_then_direct = bdRes;
      results.tests[bundleDirectTestKey] = {
        status: "PASS",
        detail: `${key}: Bundle drew Green triangle (${bdRes.green_pixel.join(",")}), Direct draw drew Blue triangle (${bdRes.blue_pixel.join(",")}) after executeBundles state clear (BG=[${bdRes.background_pixel.join(",")}])`,
        implementation_owner: owner,
      };
    } catch (err) {
      variantRecord.bundle_then_direct = { status: "FAIL", error: err.message || String(err) };
      results.tests[bundleDirectTestKey] = {
        status: "FAIL",
        error: err.message || String(err),
        implementation_owner: owner,
      };
    }

    const passedBoth =
      variantRecord.red_a_blue_b?.status === "PASS" &&
      variantRecord.bundle_then_direct?.status === "PASS";
    variantRecord.preconditions_met = passedBoth;
    if (!passedBoth) {
      results.excluded_variants.push(key);
    }

    results.variants[key] = variantRecord;
  }

  return results;
}
