/**
 * direct_reference.js - Direct-JS WebGPU Oracle Reference
 * 
 * Executes identical WebGPU rendering work directly without going through
 * the binary bridge packet decoder, for pixel-identical comparison.
 */

/**
 * Shared WGSL shader and vertex data for direct reference triangle rendering.
 */
const DIRECT_TRIANGLE_SHADER_CODE = `
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

const DIRECT_TRIANGLE_VERTEX_DATA = new Float32Array([
  // x,    y,    z,   u,   v
   0.0,  0.5,  0.0, 0.5, 1.0,
  -0.5, -0.5,  0.0, 0.0, 0.0,
   0.5, -0.5,  0.0, 1.0, 0.0,
]);

/**
 * Reusable Direct-JS WebGPU Oracle Renderer.
 *
 * Allocates GPU resources (target texture, vertex buffer, uniform buffer, pipeline,
 * bind group, and readback buffer) once, allowing synchronous repeated-frame rendering
 * for fixed drawCount transforms without per-frame GPU reallocation.
 *
 * @param {GPUDevice} device
 * @param {number} [width=64]
 * @param {number} [height=64]
 * @param {number} [drawCount=1]
 */
export function createDirectReferenceRenderer(device, width = 64, height = 64, drawCount = 1) {
  if (!Number.isInteger(drawCount) || drawCount <= 0) {
    throw new RangeError("drawCount must be a positive integer");
  }

  // 1. Offscreen target texture
  const targetTexture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // 2. Vertex buffer: 3 vertices with position (vec3) and uv (vec2)
  const vertexBuffer = device.createBuffer({
    size: DIRECT_TRIANGLE_VERTEX_DATA.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, DIRECT_TRIANGLE_VERTEX_DATA);

  // 3. AffineRows uniform buffer: ONE uniform buffer for all draws
  const uniformBuffer = device.createBuffer({
    size: drawCount * 256, // 48B records at 256B dynamic offsets
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 4. Pipeline & Bind Group
  const shaderModule = device.createShaderModule({ code: DIRECT_TRIANGLE_SHADER_CODE });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 48,
        },
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
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer, offset: 0, size: 48 },
      },
    ],
  });

  // 5. Readback buffer
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  /**
   * Synchronously submits one frame of drawCount draws and texture-to-readback copy.
   *
   * Validates data count exactly equals initialized drawCount before GPU write,
   * packs inside frame call, writes uniform then encodes same draws AND same
   * texture->readback copy as bulk update, synchronously submits, returns queue.onSubmittedWorkDone Promise.
   *
   * @param {Float32Array|null} [affineRows=null]
   * @param {Function|null} [drawEncoder=null]
   * @param {Uint8Array|Float32Array|Function|null} [preparedUniformData=null]
   * @param {Object|null} [timingRecord=null]
   * @returns {Promise<void>}
   */
  function submitFrame(affineRows = null, drawEncoder = null, preparedUniformData = null, timingRecord = null) {
    if (drawEncoder !== null && typeof drawEncoder !== "function") {
      throw new TypeError("drawEncoder must be a function or null");
    }

    let uniformData;
    if (preparedUniformData !== null) {
      const raw = typeof preparedUniformData === "function" ? preparedUniformData() : preparedUniformData;
      if (raw instanceof Uint8Array) {
        if (raw.byteLength !== drawCount * 256) {
          throw new RangeError(`preparedUniformData Uint8Array byteLength (${raw.byteLength}) must equal drawCount * 256 (${drawCount * 256})`);
        }
        uniformData = raw;
      } else if (raw instanceof Float32Array) {
        if (raw.length !== drawCount * 64) {
          throw new RangeError(`preparedUniformData Float32Array length (${raw.length}) must equal drawCount * 64 (${drawCount * 64})`);
        }
        uniformData = raw;
      } else {
        throw new RangeError("preparedUniformData must be a non-empty Uint8Array or Float32Array with 256-byte aligned records");
      }
    } else if (affineRows !== null) {
      if (!(affineRows instanceof Float32Array) || affineRows.length !== drawCount * 12) {
        throw new RangeError(`affineRows must be a Float32Array of length drawCount * 12 (${drawCount * 12})`);
      }
      uniformData = new Float32Array(drawCount * 64);
      for (let i = 0; i < drawCount; i++) {
        const srcOffset = i * 12;
        const dstOffset = i * 64;
        for (let j = 0; j < 12; j++) {
          uniformData[dstOffset + j] = affineRows[srcOffset + j];
        }
      }
    } else {
      if (drawCount !== 1) {
        throw new RangeError(`null affineRows requires drawCount = 1, but renderer was initialized with drawCount = ${drawCount}`);
      }
      uniformData = new Float32Array(64);
      uniformData[0] = 1.0;
      uniformData[5] = 1.0;
      uniformData[10] = 1.0;
    }

    // Write uniform before encode/submit
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Encode render pass and same texture->readback copy
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
    pass.setVertexBuffer(0, vertexBuffer);
    if (typeof drawEncoder === "function") {
      drawEncoder(pass, bindGroup, drawCount);
    } else {
      for (let i = 0; i < drawCount; i++) {
        pass.setBindGroup(0, bindGroup, [i * 256]);
        pass.draw(3, 1, 0, 0);
      }
    }
    pass.end();

    encoder.copyTextureToBuffer(
      { texture: targetTexture },
      { buffer: readbackBuffer, bytesPerRow: bytesPerRow, rowsPerImage: height },
      [width, height, 1]
    );

    const commandBuffer = encoder.finish();
    const tFinish = timingRecord ? performance.now() : 0;
    const tSubmitStart = timingRecord ? performance.now() : 0;
    device.queue.submit([commandBuffer]);
    const tSubmitEnd = timingRecord ? performance.now() : 0;
    if (timingRecord) {
      timingRecord._tFinish = tFinish;
      timingRecord._tSubmitStart = tSubmitStart;
      timingRecord._tSubmitEnd = tSubmitEnd;
    }

    return device.queue.onSubmittedWorkDone();
  }

  async function readback() {
    await readbackBuffer.mapAsync(GPUMapMode.READ, 0, bytesPerRow * height);
    const mapped = readbackBuffer.getMappedRange(0, bytesPerRow * height);
    const result = new Uint8Array(mapped.slice(0));
    readbackBuffer.unmap();
    return result;
  }

  function destroy() {
    targetTexture.destroy();
    vertexBuffer.destroy();
    uniformBuffer.destroy();
    readbackBuffer.destroy();
  }

  return {
    device,
    width,
    height,
    drawCount,
    targetTexture,
    vertexBuffer,
    uniformBuffer,
    pipeline,
    bindGroup,
    readbackBuffer,
    submitFrame,
    readback,
    destroy,
  };
}

export async function renderDirectReferenceTriangle(
  device,
  width = 64,
  height = 64,
  measurementSeam = null,
  affineRows = null,
  drawEncoder = null,
  preparedUniformData = null
) {
  if (drawEncoder !== null && typeof drawEncoder !== "function") {
    throw new TypeError("drawEncoder must be a function or null");
  }

  const measure = measurementSeam !== null && typeof measurementSeam === "object";
  const t0 = measure ? performance.now() : 0;

  let drawCount = 1;
  let resolvedUniformData = null;

  if (preparedUniformData !== null) {
    const raw = typeof preparedUniformData === "function" ? preparedUniformData() : preparedUniformData;
    if (raw instanceof Uint8Array) {
      if (raw.byteLength === 0 || raw.byteLength % 256 !== 0) {
        throw new RangeError("preparedUniformData Uint8Array byteLength must be non-empty and divisible by 256");
      }
      drawCount = raw.byteLength / 256;
      resolvedUniformData = raw;
    } else if (raw instanceof Float32Array) {
      if (raw.length === 0 || raw.length % 64 !== 0) {
        throw new RangeError("preparedUniformData Float32Array length must be non-empty and divisible by 64");
      }
      drawCount = raw.length / 64;
      resolvedUniformData = raw;
    } else {
      throw new RangeError("preparedUniformData must be a non-empty Uint8Array or Float32Array with 256-byte aligned records");
    }
  } else if (affineRows !== null) {
    if (!(affineRows instanceof Float32Array) || affineRows.length === 0 || affineRows.length % 12 !== 0) {
      throw new RangeError("affineRows must be a non-empty Float32Array with length divisible by 12");
    }
    drawCount = affineRows.length / 12;
  }

  const renderer = createDirectReferenceRenderer(device, width, height, drawCount);
  try {
    const timingRecord = measure ? {} : null;
    const donePromise = renderer.submitFrame(
      affineRows,
      drawEncoder,
      resolvedUniformData || preparedUniformData,
      timingRecord
    );

    if (measure) {
      const tSubmitEnd = timingRecord._tSubmitEnd;
      measurementSeam.direct_prepare_ms = timingRecord._tFinish - t0;
      measurementSeam.direct_submit_ms = tSubmitEnd - timingRecord._tSubmitStart;
      measurementSeam.cpu_prepare_submit_ms = measurementSeam.direct_prepare_ms + measurementSeam.direct_submit_ms;
      await donePromise;
      measurementSeam.gpu_complete_ms = performance.now() - tSubmitEnd;
    } else {
      await donePromise;
    }

    return await renderer.readback();
  } finally {
    renderer.destroy();
  }
}

/**
 * Direct-JS WebGPU Oracle Reference: Textured Triangle with AffineRows Transform.
 *
 * Executes identical WebGPU rendering work using pure direct WebGPU API calls
 * (own texture, sampler, uniform buffer, and pipeline), without going through
 * the binary bridge packet decoder.
 *
 * @param {GPUDevice} device
 * @param {Uint8Array} pixels - Raw texture pixels (RGBA8)
 * @param {number} [texW=2] - Source texture width
 * @param {number} [texH=2] - Source texture height
 * @param {Float32Array|number[]} [affine=null] - 12 floats representing AffineRows
 * @param {number} [width=64] - Target width
 * @param {number} [height=64] - Target height
 * @returns {Promise<Uint8Array>}
 */
export async function renderDirectReferenceTexturedTriangle(
  device,
  pixels,
  texW = 2,
  texH = 2,
  affine = null,
  width = 64,
  height = 64
) {
  const shaderCode = `
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

@group(0) @binding(1)
var tex: texture_2d<f32>;

@group(0) @binding(2)
var samp: sampler;

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
    return textureSample(tex, samp, in.uv);
}
`;

  // 1. Offscreen target texture
  const targetTexture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // 2. Source texture to sample from
  const sourceTexture = device.createTexture({
    size: [texW, texH, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.writeTexture(
    { texture: sourceTexture },
    pixels,
    { bytesPerRow: texW * 4, rowsPerImage: texH },
    [texW, texH, 1]
  );

  // 3. Sampler: nearest filter, clamp-to-edge
  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  // 4. Vertex buffer: 3 vertices matching gpu_host.rs AffineRows triangle
  // Vertex 0: pos (0.0, 0.5, 0.0), uv (0.5, 1.0)
  // Vertex 1: pos (-0.5, -0.5, 0.0), uv (0.0, 0.0)
  // Vertex 2: pos (0.5, -0.5, 0.0), uv (1.0, 0.0)
  const vertexData = new Float32Array([
     0.0,  0.5,  0.0, 0.5, 1.0,
    -0.5, -0.5,  0.0, 0.0, 0.0,
     0.5, -0.5,  0.0, 1.0, 0.0,
  ]);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // 5. AffineRows uniform buffer (48 bytes: 3 rows of vec4)
  const affineData = new Float32Array(
    affine || [
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
    ]
  );
  const uniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, affineData);

  // 6. Pipeline layout & bind group layout
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 48,
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {
          type: "filtering",
        },
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
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer, offset: 0, size: 48 },
      },
      {
        binding: 1,
        resource: sourceTexture.createView(),
      },
      {
        binding: 2,
        resource: sampler,
      },
    ],
  });

  // 7. Readback buffer
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const readbackBuffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // 8. Encode and submit
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

  // 9. Readback
  await readbackBuffer.mapAsync(GPUMapMode.READ, 0, bytesPerRow * height);
  const mapped = readbackBuffer.getMappedRange(0, bytesPerRow * height);
  const result = new Uint8Array(mapped.slice(0));
  readbackBuffer.unmap();

  return result;
}


import {
  PACKET_MAGIC,
  PACKET_VERSION,
  OPCODE_CREATE_BUFFER,
  OPCODE_WRITE_BUFFER,
  OPCODE_CREATE_TEXTURE,
  OPCODE_CREATE_PIPELINE,
  OPCODE_RENDER_PASS,
  OPCODE_COPY_TEXTURE_TO_BUFFER,
  OPCODE_RECORD_BUNDLE,
  OPCODE_EXECUTE_BUNDLES,
  TEXTURE_USAGE_COPY_SRC,
  TEXTURE_USAGE_RENDER_ATTACHMENT,
} from "./bridge_runtime.js";

export { TEXTURE_USAGE_COPY_SRC, TEXTURE_USAGE_RENDER_ATTACHMENT, OPCODE_RECORD_BUNDLE, OPCODE_EXECUTE_BUNDLES };

/**
 * Independent JS-side PacketBuilder kept exclusively inside direct_reference.js
 * as an oracle reference implementation of the binary packet format.
 */
export class PacketBuilder {
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

    let dataCursor = headerLen + commandBytesLen;
    for (const chunk of this.dataChunks) {
      out.set(chunk, dataCursor);
      dataCursor += chunk.byteLength;
    }

    return out;
  }
}
