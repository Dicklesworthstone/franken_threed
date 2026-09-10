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
export async function renderDirectBundleDirectReference(device, width = 32, height = 32) {
  const target = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const shaderModule = device.createShaderModule({ code: WGSL_SOLID_COLOR });
  const bgl = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: { module: shaderModule, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });

  // Uniform buffer:
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

  // Record a RenderBundle that draws with Offset 0 (Green)
  const bundleEncoder = device.createRenderBundleEncoder({
    colorFormats: ["rgba8unorm"],
  });
  bundleEncoder.setPipeline(pipeline);
  bundleEncoder.setBindGroup(0, bindGroup, [0]);
  bundleEncoder.draw(6, 1, 0, 0);
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

  // Step 1: Pre-set pipeline and bind group with Offset 256 (Blue)
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup, [256]);

  // Step 2: Execute bundle (draws Green). Under WebGPU spec, executeBundles
  // resets active pipeline, bind groups, and vertex/index buffers.
  pass.executeBundles([bundle]);

  // Step 3: Execute an empty bundle sequence (spec mandates this also resets state)
  pass.executeBundles([]);

  // Step 4: Direct draw. Because executeBundles cleared state, adapter MUST re-bind
  // pipeline and bind group. If not re-bound, WebGPU throws validation error or fails.
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup, [256]);
  pass.draw(6, 1, 0, 0);

  pass.end();

  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height, 1]);
  device.queue.submit([encoder.finish()]);

  const pixels = await readbackGpuBuffer(device, readback, bytesPerRow * height);

  target.destroy();
  uniformBuffer.destroy();
  readback.destroy();

  return pixels;
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
