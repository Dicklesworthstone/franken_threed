/**
 * direct_reference.js - Direct-JS WebGPU Oracle Reference
 * 
 * Executes identical WebGPU rendering work directly without going through
 * the binary bridge packet decoder, for pixel-identical comparison.
 */

export async function renderDirectReferenceTriangle(device, width = 64, height = 64) {
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

  // 1. Offscreen target texture
  const targetTexture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // 2. Vertex buffer: 3 vertices with position (vec3) and uv (vec2)
  // Triangle in NDC [-0.5, 0.5]
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

  // 3. AffineRows uniform buffer: Identity transform (48 bytes: 3 rows of vec4)
  // Row 0: [1, 0, 0, 0]
  // Row 1: [0, 1, 0, 0]
  // Row 2: [0, 0, 1, 0]
  const affineData = new Float32Array([
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
  ]);
  const uniformBuffer = device.createBuffer({
    size: 256, // aligned to minUniformBufferOffsetAlignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, affineData);

  // 4. Pipeline
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

  // 6. Encode and submit
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
    { buffer: readbackBuffer, bytesPerRow: bytesPerRow, rowsPerImage: height },
    [width, height, 1]
  );

  device.queue.submit([encoder.finish()]);

  // 7. Readback
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
