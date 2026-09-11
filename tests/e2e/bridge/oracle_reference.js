/**
 * oracle_reference.js - Independent Direct-JavaScript WebGPU Oracle Reference
 * 
 * Bead: f3d-05-ids-layouts-epochs-transport-vqa.7
 * 
 * Provides independent direct-JS WebGPU reference implementations that execute
 * standard WebGPU API calls directly without relying on candidate binary bridge
 * packet decoders, serializers, or higher-level abstractions.
 * 
 * Used as the ground-truth oracle for differential pixel and state readback.
 */

/**
 * Procedural WGSL shader using AffineRows (48 bytes) for transformation.
 */
export const WGSL_AFFINE_TRIANGLE = `
struct AffineRows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
};

fn transform_affine_point(m: AffineRows, p: vec3<f32>) -> vec3<f32> {
    let v = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

@group(0) @binding(0)
var<uniform> model: AffineRows;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let transformed = transform_affine_point(model, in.position);
    out.clip_pos = vec4<f32>(transformed, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv.x, in.uv.y, 1.0 - in.uv.x, 1.0);
}
`;

/**
 * WGSL shader for solid color quad rendering with dynamic uniform buffer offset.
 */
export const WGSL_SOLID_COLOR = `
struct ColorUniform {
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uColor: ColorUniform;

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen / NDC quad from 6 vertices
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    return vec4<f32>(pos[vertex_idx], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uColor.color;
}
`;

/**
 * WGSL shader for flat color uniform with VertexPosUv layout (matches gpu_host.rs).
 */
export const WGSL_FLAT_COLOR_BUNDLE = `
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

/**
 * Helper to compute WebGPU row-pitch aligned to 256 bytes.
 */
export function computeAlignedBytesPerRow(width) {
  const unaligned = width * 4;
  return Math.ceil(unaligned / 256) * 256;
}

/**
 * Reads back a GPUBuffer into a new Uint8Array.
 */
export async function readbackGpuBuffer(device, buffer, byteLength) {
  await buffer.mapAsync(GPUMapMode.READ, 0, byteLength);
  const mapped = buffer.getMappedRange(0, byteLength);
  const copy = new Uint8Array(mapped.slice(0));
  buffer.unmap();
  return copy;
}

/**
 * 1. Independent Direct-JS Reference: First-Frame Triangle
 * Renders an AffineRows-transformed triangle directly to an offscreen texture.
 */
export async function renderDirectTriangleReference(device, width = 64, height = 64) {
  const targetTexture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // Vertex buffer: 3 vertices with position (vec3) and uv (vec2)
  const vertexData = new Float32Array([
    // x,    y,    z,   u,   v
     0.0,  0.5,  0.0, 0.5, 1.0,
    -0.5, -0.5,  0.0, 0.0, 0.0,
     0.5, -0.5,  0.0, 1.0, 0.0,
  ]);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // Identity AffineRows: 48 bytes (3 rows of vec4)
  const affineData = new Float32Array([
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
  ]);
  const uniformBuffer = device.createBuffer({
    size: 256, // Aligned to minUniformBufferOffsetAlignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, affineData);

  const shaderModule = device.createShaderModule({ code: WGSL_AFFINE_TRIANGLE });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 48 },
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
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: "rgba8unorm" }],
    },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: 0, size: 48 } }],
  });

  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: targetTexture.createView(),
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup, [0]);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(3, 1, 0, 0);
  pass.end();

  encoder.copyTextureToBuffer(
    { texture: targetTexture },
    { buffer: readbackBuffer, bytesPerRow, rowsPerImage: height },
    [width, height, 1]
  );

  device.queue.submit([encoder.finish()]);
  const pixels = await readbackGpuBuffer(device, readbackBuffer, bytesPerRow * height);

  targetTexture.destroy();
  vertexBuffer.destroy();
  uniformBuffer.destroy();
  readbackBuffer.destroy();

  return pixels;
}

/**
 * 2. Independent Direct-JS Reference: Red-A / Blue-B Queue-Write Snapshot Isolation
 * Renders Red to Target A and Blue to Target B in a SINGLE queue submission using
 * versioned buffer slices (offset 0 and offset 256).
 */
export async function renderDirectRedABlueBReference(device, width = 32, height = 32) {
  const targetA = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const targetB = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // Uniform buffer with 2 versioned slices of 256 bytes each:
  // Slice 0 (offset 0): Red   [1.0, 0.0, 0.0, 1.0]
  // Slice 1 (offset 256): Blue [0.0, 0.0, 1.0, 1.0]
  const uniformBuffer = device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const redData = new Float32Array([1.0, 0.0, 0.0, 1.0]);
  const blueData = new Float32Array([0.0, 0.0, 1.0, 1.0]);
  device.queue.writeBuffer(uniformBuffer, 0, redData);
  device.queue.writeBuffer(uniformBuffer, 256, blueData);

  const shaderModule = device.createShaderModule({ code: WGSL_SOLID_COLOR });
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

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: 0, size: 16 } }],
  });

  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readbackA = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const readbackB = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();

  // Pass A -> Target A using dynamic offset 0 (Red)
  const passA = encoder.beginRenderPass({
    colorAttachments: [{
      view: targetA.createView(),
      clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  passA.setPipeline(pipeline);
  passA.setBindGroup(0, bindGroup, [0]);
  passA.draw(6, 1, 0, 0);
  passA.end();

  // Pass B -> Target B using dynamic offset 256 (Blue)
  const passB = encoder.beginRenderPass({
    colorAttachments: [{
      view: targetB.createView(),
      clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  passB.setPipeline(pipeline);
  passB.setBindGroup(0, bindGroup, [256]);
  passB.draw(6, 1, 0, 0);
  passB.end();

  encoder.copyTextureToBuffer({ texture: targetA }, { buffer: readbackA, bytesPerRow, rowsPerImage: height }, [width, height, 1]);
  encoder.copyTextureToBuffer({ texture: targetB }, { buffer: readbackB, bytesPerRow, rowsPerImage: height }, [width, height, 1]);

  // Single submission for both passes
  device.queue.submit([encoder.finish()]);

  const pixelsA = await readbackGpuBuffer(device, readbackA, bytesPerRow * height);
  const pixelsB = await readbackGpuBuffer(device, readbackB, bytesPerRow * height);

  targetA.destroy();
  targetB.destroy();
  uniformBuffer.destroy();
  readbackA.destroy();
  readbackB.destroy();

  return { pixelsA, pixelsB };
}

/**
 * 3. Independent Direct-JS Reference: Bundle-Then-Direct-Draw State Reset
 * Demonstrates that executeBundles clears render-pass state, requiring explicit
 * re-binding before subsequent direct draws.
 */
export async function renderDirectBundleDirectReference(device, width = 64, height = 64) {
  const target = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const shaderModule = device.createShaderModule({ code: WGSL_FLAT_COLOR_BUNDLE });
  const bgl = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
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

  // Uniform buffer (512 bytes):
  // Offset 0: Green [0.0, 1.0, 0.0, 1.0] (recorded in bundle)
  // Offset 256: Blue [0.0, 0.0, 1.0, 1.0] (drawn in direct draw)
  const uniformBuffer = device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([0.0, 1.0, 0.0, 1.0]));
  device.queue.writeBuffer(uniformBuffer, 256, new Float32Array([0.0, 0.0, 1.0, 1.0]));

  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: 0, size: 16 } }],
  });

  // Vertex buffer 1: Triangle 1 (left side)
  const tri1Data = new Float32Array([
    -1.0,  1.0, 0.0,  0.0, 1.0,
    -1.0, -1.0, 0.0,  0.0, 0.0,
     0.0,  1.0, 0.0,  0.5, 1.0,
  ]);
  const vb1 = device.createBuffer({
    size: tri1Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb1, 0, tri1Data);

  // Vertex buffer 2: Triangle 2 (right side)
  const tri2Data = new Float32Array([
     0.0, -1.0, 0.0,  0.5, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
  ]);
  const vb2 = device.createBuffer({
    size: tri2Data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb2, 0, tri2Data);

  // Record a RenderBundle that draws Triangle 1 with Offset 0 (Green)
  const bundleEncoder = device.createRenderBundleEncoder({
    colorFormats: ["rgba8unorm"],
  });
  bundleEncoder.setPipeline(pipeline);
  bundleEncoder.setBindGroup(0, bindGroup, [0]);
  bundleEncoder.setVertexBuffer(0, vb1);
  bundleEncoder.draw(3, 1, 0, 0);
  const bundle = bundleEncoder.finish();

  const bytesPerRow = computeAlignedBytesPerRow(width);
  const readback = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: target.createView(),
      clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });

  // Step 1: Execute bundle (draws Green left triangle).
  pass.executeBundles([bundle]);

  // Step 2: Execute empty bundle sequence (spec mandates this also resets state).
  pass.executeBundles([]);

  // Step 3: Direct draw Triangle 2. Because executeBundles cleared state, adapter
  // MUST re-bind pipeline, bind group (offset 256 = Blue), and vertex buffer (vb2).
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup, [256]);
  pass.setVertexBuffer(0, vb2);
  pass.draw(3, 1, 0, 0);

  pass.end();

  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height, 1]);
  device.queue.submit([encoder.finish()]);

  const pixels = await readbackGpuBuffer(device, readback, bytesPerRow * height);

  target.destroy();
  uniformBuffer.destroy();
  vb1.destroy();
  vb2.destroy();
  readback.destroy();

  return pixels;
}

/**
 * Dual Assertion Helper: Bundle-Then-Direct-Draw Verification.
 * Asserts pixel-for-pixel identity against independent direct-JS oracle
 * and verifies expected color values at designated sample points:
 * - Left side (x=16, y=32): Green [0, 255, 0, 255]
 * - Right side (x=48, y=32): Blue [0, 0, 255, 255]
 * - Background (x=2, y=2): Black [0, 0, 0, 255]
 */
export function assertBundleDirectDrawMatch(candidatePixels, oraclePixels, width = 64, height = 64) {
  if (candidatePixels.byteLength !== oraclePixels.byteLength) {
    throw new Error(`assertBundleDirectDrawMatch: byte length mismatch (candidate=${candidatePixels.byteLength}, oracle=${oraclePixels.byteLength})`);
  }
  let diffCount = 0;
  for (let i = 0; i < candidatePixels.length; i++) {
    if (candidatePixels[i] !== oraclePixels[i]) {
      diffCount++;
    }
  }
  if (diffCount > 0) {
    throw new Error(`assertBundleDirectDrawMatch: detected ${diffCount} mismatched bytes out of ${candidatePixels.length}`);
  }

  const bytesPerRow = computeAlignedBytesPerRow(width);

  // 1. Left side (x=16, y=32): Green [0, 255, 0, 255] from bundle draw
  const greenIdx = 32 * bytesPerRow + 16 * 4;
  const gr = candidatePixels[greenIdx];
  const gg = candidatePixels[greenIdx + 1];
  const gb = candidatePixels[greenIdx + 2];
  const ga = candidatePixels[greenIdx + 3];
  if (gr > 5 || gg < 250 || gb > 5 || ga < 250) {
    throw new Error(
      `Bundle-Then-Direct Draw sample violation at (16, 32): expected Green [0, 255, 0, 255], observed [${gr}, ${gg}, ${gb}, ${ga}]`
    );
  }

  // 2. Right side (x=48, y=32): Blue [0, 0, 255, 255] from direct draw after state reset
  const blueIdx = 32 * bytesPerRow + 48 * 4;
  const br = candidatePixels[blueIdx];
  const bg = candidatePixels[blueIdx + 1];
  const bb = candidatePixels[blueIdx + 2];
  const ba = candidatePixels[blueIdx + 3];
  if (br > 5 || bg > 5 || bb < 250 || ba < 250) {
    throw new Error(
      `Bundle-Then-Direct Draw sample violation at (48, 32): expected Blue [0, 0, 255, 255], observed [${br}, ${bg}, ${bb}, ${ba}]`
    );
  }

  // 3. Background (x=2, y=2): Black [0, 0, 0, 255] clear color
  const blackIdx = 2 * bytesPerRow + 2 * 4;
  const kr = candidatePixels[blackIdx];
  const kg = candidatePixels[blackIdx + 1];
  const kb = candidatePixels[blackIdx + 2];
  const ka = candidatePixels[blackIdx + 3];
  if (kr > 5 || kg > 5 || kb > 5 || ka < 250) {
    throw new Error(
      `Bundle-Then-Direct Draw sample violation at (2, 2): expected Black [0, 0, 0, 255], observed [${kr}, ${kg}, ${kb}, ${ka}]`
    );
  }
}

/**
 * Pure CPU algebraic transform for AffineRows dot product.
 */
export function evalDirectAffineTransform(matrixColumnMajor, point) {
  const e = matrixColumnMajor;
  // AffineRows definition:
  // r0: [e0, e4, e8,  e12]
  // r1: [e1, e5, e9,  e13]
  // r2: [e2, e6, e10, e14]
  const v = [point[0], point[1], point[2], 1.0];
  const x = e[0] * v[0] + e[4] * v[1] + e[8]  * v[2] + e[12] * v[3];
  const y = e[1] * v[0] + e[5] * v[1] + e[9]  * v[2] + e[13] * v[3];
  const z = e[2] * v[0] + e[6] * v[1] + e[10] * v[2] + e[14] * v[3];
  return [x, y, z];
}

/**
 * Dual Assertion Helper: Generational Handle Publication (§6.5, vqa.6).
 * Asserts that a resource handle index and generation are active and fresh in the
 * bridge slot table. If the handle is stale (older generation after release/reuse),
 * unallocated, or generation 0, rejects the publication attempt.
 */
export function assertGenerationalHandlePublication(checkFn, resourceId, generation) {
  if (typeof checkFn !== "function") {
    throw new Error("assertGenerationalHandlePublication: checkFn must be a function");
  }
  const isValid = checkFn(resourceId, generation);
  if (!isValid) {
    throw new Error(
      `Generational Handle Publication Rejected: Resource handle (id=${resourceId}, generation=${generation}) is invalid, stale, or revoked!`
    );
  }
  return true;
}

/**
 * Dual Assertion Helper: Linear Memory Growth Invariant (§6.6, §13.1, vqa.6).
 * Asserts that linear memory growth is permitted. If an open borrow scope is active
 * across linear memory, growth is strictly refused to prevent pointer/view invalidation.
 */
export function assertMemoryGrowthAllowed(growFn, pages = 1) {
  if (typeof growFn !== "function") {
    throw new Error("assertMemoryGrowthAllowed: growFn must be a function");
  }
  const allowed = growFn(pages);
  if (!allowed) {
    throw new Error(
      `Linear Memory Borrow Violation: Attempted linear memory growth (${pages} pages) while an active borrow scope is held!`
    );
  }
  return true;
}

/**
 * Dual Assertion Helper: AffineRows GPU Wire Layout Validation (§6.1, §6.2, vqa.6).
 * Validates a matrix or wire buffer payload against AffineRows invariants.
 * Throws structured rejection if buffer is smaller than 48 bytes (code 1),
 * contains non-affine perspective elements or invalid scaling (code 2),
 * or encounters other layout violations (codes 3..7).
 */
export function assertAffineRowsLayoutValid(validateFn, bytes) {
  if (typeof validateFn !== "function") {
    throw new Error("assertAffineRowsLayoutValid: validateFn must be a function");
  }
  const code = validateFn(bytes);
  if (code !== 0) {
    const reason = code === 1
      ? "BufferTooSmall (length < 48 or 49..63 bytes)"
      : code === 2
      ? "NonAffineMatrix (perspective elements non-zero, invalid scale, or non-finite)"
      : code === 4
      ? "IncompatibleTargetLayout (length > 64 bytes)"
      : `LayoutError code ${code}`;
    throw new Error(`AffineRows Layout Validation Rejected: ${reason} (code=${code})`);
  }
  return true;
}


