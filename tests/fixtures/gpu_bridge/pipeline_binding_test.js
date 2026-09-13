import {
  PACKET_MAGIC, PACKET_VERSION, OPCODE_RENDER_PASS,
  OPCODE_COPY_TEXTURE_TO_BUFFER, OPCODE_WRITE_TEXTURE,
  OPCODE_CREATE_PIPELINE_TEXTURED, TARGET_OFFSCREEN,
} from "./bridge_runtime.js";
import { PacketBuilder } from "./direct_reference.js";

const COLOR_SHADER_CODE = `
  @group(0) @binding(0) var<uniform> color: vec4<f32>;
  @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
    return vec4(p[i], 0., 1.);
  }
  @fragment fn fs_main() -> @location(0) vec4<f32> { return color; }
`;

function assertPixel(bytes, expected, label) {
  if (expected.some((value, index) => bytes[index] !== value)) {
    throw new Error(`${label}: expected ${Array.from(expected)}, got ${Array.from(bytes.slice(0, 4))}`);
  }
}

// Exercise the production JS decoder with real incompatible pipeline layouts.
export async function testPipelineBindingChange(host) {
  const uniformId = 0x7fffff00, targetId = 0x7fffff01, readbackId = 0x7fffff02;
  const pipelineIds = [0x7fffff03, 0x7fffff04];
  const device = host.device;
  const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const target = device.createTexture({
    size: [1, 1], format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  try {
    await host.withErrorScopes(["validation"], () => {
      device.queue.writeBuffer(uniform, 0, new Float32Array([0, 1, 0, 1, 0, 0, 0, 0]));
      const module = device.createShaderModule({ code: COLOR_SHADER_CODE });
      for (const [index, uniformSize] of [16, 32].entries()) {
        const bindGroupLayout = device.createBindGroupLayout({ entries: [{
          binding: 0, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: uniformSize },
        }] });
        const pipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
          vertex: { module, entryPoint: "vs_main" },
          fragment: { module, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
          primitive: { topology: "triangle-list" },
        });
        host.pipelines.set(pipelineIds[index], { pipeline, bindGroupLayout, hasUniformBuffer: true, uniformSize });
      }
    });
    host.buffers.set(uniformId, uniform);
    host.buffers.set(readbackId, readback);
    host.textures.set(targetId, target);

    // Two draws in one pass, same uniform buffer and offset, different layouts.
    // The original cache reused the 16-byte binding for the 32-byte layout.
    const bytes = new Uint8Array(16 + 2 * 46 + 26);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, PACKET_MAGIC, true);
    view.setUint16(4, PACKET_VERSION, true);
    view.setUint32(8, 3, true);
    let cursor = 16;
    for (const pipelineId of pipelineIds) {
      view.setUint16(cursor, OPCODE_RENDER_PASS, true);
      cursor += 2;
      view.setUint32(cursor, TARGET_OFFSCREEN, true);
      view.setUint32(cursor + 4, targetId, true);
      view.setFloat32(cursor + 20, 1, true);
      view.setUint32(cursor + 24, pipelineId, true);
      view.setUint32(cursor + 32, 3, true);
      view.setUint32(cursor + 40, uniformId, true);
      cursor += 44;
    }
    view.setUint16(cursor, OPCODE_COPY_TEXTURE_TO_BUFFER, true);
    cursor += 2;
    [targetId, readbackId, 1, 1, 0, 0].forEach((n, i) => view.setUint32(cursor + i * 4, n, true));
    await host.executePacket(bytes);
    const pixel = await host.readbackBuffer(readbackId, 256);
    if (pixel[0] !== 0 || pixel[1] !== 255 || pixel[2] !== 0 || pixel[3] !== 255) {
      throw new Error(`Pipeline binding switch rendered wrong pixel: ${Array.from(pixel.slice(0, 4))}`);
    }
    await testTextureBindingAndWriteOrder(host);
    await testSameIdBufferReplacement(host);
    return "JS host bridge: pipeline layout switch renders green; texture-only pipeline renders blue; interleaved texture writes preserve red-A/blue-B; same-ID buffer replacement renders red-A/blue-B";
  } finally {
    host.buffers.delete(uniformId);
    host.buffers.delete(readbackId);
    host.bufferEpochs.delete(readbackId);
    host.textures.delete(targetId);
    for (const id of pipelineIds) host.pipelines.delete(id);
    uniform.destroy();
    readback.destroy();
    target.destroy();
  }
}

async function testTextureBindingAndWriteOrder(host) {
  const sourceId = 0x7fffff10, targetId = 0x7fffff11, pipelineId = 0x7fffff12;
  const readbackIds = [0x7fffff13, 0x7fffff14];
  const previousTargetId = host.lastRenderTargetId;
  const red = new Uint8Array([255, 0, 0, 255]);
  const blue = new Uint8Array([0, 0, 255, 255]);
  const writeCommandBytes = 26, pipelineCommandBytes = 46;

  // Only the two new opcodes need local encoding; all existing commands use
  // the fixture's PacketBuilder. These uploads are exactly one RGBA8 texel.
  function writeTextureCommand(view, cursor, dataOffset) {
    view.setUint16(cursor, OPCODE_WRITE_TEXTURE, true);
    [sourceId, 1, 1, 4, dataOffset, 4].forEach((value, index) =>
      view.setUint32(cursor + 2 + index * 4, value, true));
  }

  try {
    const setup = new PacketBuilder();
    setup.createTexture(sourceId, 1, 1, 2,
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC);
    setup.createTexture(targetId, 1, 1, 2,
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
    for (const id of readbackIds) {
      setup.createBuffer(id, 256, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    }
    await host.executePacket(setup.build());

    const shader = new TextEncoder().encode(`
      @group(0) @binding(1) var tex: texture_2d<f32>;
      @group(0) @binding(2) var samp: sampler;
      @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
        let p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
        return vec4(p[i], 0., 1.);
      }
      @fragment fn fs_main() -> @location(0) vec4<f32> {
        return textureSample(tex, samp, vec2<f32>(0.5));
      }
    `);
    const draw = new PacketBuilder();
    draw.renderPass(TARGET_OFFSCREEN, targetId, [0, 0, 0, 1], pipelineId, 0, 3);
    draw.copyTextureToBuffer(targetId, readbackIds[0], 1, 1);
    const drawPacket = draw.build();
    const commandPrefixBytes = pipelineCommandBytes + writeCommandBytes;
    const samplePacket = new Uint8Array(drawPacket.byteLength + commandPrefixBytes + shader.byteLength + blue.byteLength);
    samplePacket.set(drawPacket.subarray(0, 16));
    const sampleView = new DataView(samplePacket.buffer);
    sampleView.setUint32(8, 4, true);
    sampleView.setUint32(12, shader.byteLength + blue.byteLength, true);
    sampleView.setUint16(16, OPCODE_CREATE_PIPELINE_TEXTURED, true);
    // has_vertex_buffer=false, has_uniform_buffer=false: bindings 1 and 2
    // must still be bound, without a fabricated binding 0 or dynamic offset.
    [pipelineId, 0, shader.byteLength, 2, 0, 0, 0, 0, sourceId, 0, 0].forEach((value, index) =>
      sampleView.setUint32(18 + index * 4, value, true));
    writeTextureCommand(sampleView, 16 + pipelineCommandBytes, shader.byteLength);
    samplePacket.set(drawPacket.subarray(16), 16 + commandPrefixBytes);
    const sampleDataStart = drawPacket.byteLength + commandPrefixBytes;
    samplePacket.set(shader, sampleDataStart);
    samplePacket.set(blue, sampleDataStart + shader.byteLength);
    await host.executePacket(samplePacket);
    assertPixel(await host.readbackBuffer(readbackIds[0], 256), blue,
      "Texture-only pipeline must sample blue without a uniform buffer");

    // A single packet must preserve command order across queue uploads and
    // encoded copies: red -> copy A -> blue -> copy B. Ending a pass alone
    // cannot order the earlier copy before the second queue.writeTexture.
    // Source and readback A start blue, so omitting the red upload or copy A
    // cannot accidentally satisfy the expected red result.
    const copies = new PacketBuilder();
    for (const id of readbackIds) copies.copyTextureToBuffer(sourceId, id, 1, 1);
    const copyPacket = copies.build();
    const copyCommandBytes = 26;
    const orderedPacket = new Uint8Array(copyPacket.byteLength + 2 * writeCommandBytes + 8);
    orderedPacket.set(copyPacket.subarray(0, 16));
    const orderedView = new DataView(orderedPacket.buffer);
    orderedView.setUint32(8, 4, true);
    orderedView.setUint32(12, 8, true);
    let cursor = 16;
    for (let index = 0; index < 2; index++) {
      writeTextureCommand(orderedView, cursor, index * 4);
      cursor += writeCommandBytes;
      const copyStart = 16 + index * copyCommandBytes;
      orderedPacket.set(copyPacket.subarray(copyStart, copyStart + copyCommandBytes), cursor);
      cursor += copyCommandBytes;
    }
    orderedPacket.set(red, cursor);
    orderedPacket.set(blue, cursor + 4);
    await host.executePacket(orderedPacket);
    assertPixel(await host.readbackBuffer(readbackIds[0], 256), red,
      "First encoded texture copy must retain red after the later blue upload");
    assertPixel(await host.readbackBuffer(readbackIds[1], 256), blue,
      "Second encoded texture copy must observe blue");
  } finally {
    for (const id of readbackIds) {
      host.buffers.get(id)?.destroy();
      host.buffers.delete(id);
      host.bufferEpochs.delete(id);
    }
    for (const id of [sourceId, targetId]) {
      host.textures.get(id)?.destroy();
      host.textures.delete(id);
    }
    host.pipelines.delete(pipelineId);
    host.lastRenderTargetId = previousTargetId;
  }
}

async function testSameIdBufferReplacement(host) {
  const uniformId = 0x7fffff20, targetId = 0x7fffff21, pipelineId = 0x7fffff22;
  const readbackAId = 0x7fffff23, readbackBId = 0x7fffff24;
  const previousTargetId = host.lastRenderTargetId;

  let oldBuffer = null;
  try {
    const setup = new PacketBuilder();
    setup.createTexture(targetId, 1, 1, 2,
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
    for (const id of [readbackAId, readbackBId]) {
      setup.createBuffer(id, 256, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    }
    setup.createBuffer(uniformId, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    setup.writeBuffer(uniformId, 0, new Uint8Array(new Float32Array([1, 0, 0, 1, 0, 0, 0, 0]).buffer));
    setup.createPipeline(pipelineId, COLOR_SHADER_CODE, 2, false, true, 32);
    await host.executePacket(setup.build());

    oldBuffer = host.buffers.get(uniformId);
    if (!oldBuffer) {
      throw new Error("Setup failed to allocate initial uniform buffer");
    }

    // Exercise one executePacket:
    // 1. Render and copy A with old uniform buffer (red).
    // 2. Create replacement buffer under SAME numeric uniform ID.
    // 3. Write distinct color (blue) to replacement buffer.
    // 4. Render and copy B using SAME pipelineRecord (must bind new buffer, not cached old buffer).
    const packet = new PacketBuilder();
    packet.renderPass(TARGET_OFFSCREEN, targetId, [0, 0, 0, 1], pipelineId, 0, 3, 0, uniformId);
    packet.copyTextureToBuffer(targetId, readbackAId, 1, 1);
    packet.createBuffer(uniformId, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    packet.writeBuffer(uniformId, 0, new Uint8Array(new Float32Array([0, 0, 1, 1, 0, 0, 0, 0]).buffer));
    packet.renderPass(TARGET_OFFSCREEN, targetId, [0, 0, 0, 1], pipelineId, 0, 3, 0, uniformId);
    packet.copyTextureToBuffer(targetId, readbackBId, 1, 1);
    await host.executePacket(packet.build());

    const newBuffer = host.buffers.get(uniformId);
    if (!newBuffer || oldBuffer === newBuffer) {
      throw new Error("Expected distinct GPUBuffer instances for same numeric ID replacement");
    }

    const pixelA = await host.readbackBuffer(readbackAId, 256);
    const pixelB = await host.readbackBuffer(readbackBId, 256);
    assertPixel(pixelA, [255, 0, 0, 255], "First pass must render red with old uniform buffer");
    assertPixel(pixelB, [0, 0, 255, 255], "Second pass must render blue with replacement uniform buffer");
  } finally {
    const currentBuffer = host.buffers.get(uniformId);
    if (currentBuffer && currentBuffer !== oldBuffer) {
      currentBuffer.destroy();
    }
    oldBuffer?.destroy();
    host.buffers.delete(uniformId);
    for (const id of [readbackAId, readbackBId]) {
      host.buffers.get(id)?.destroy();
      host.buffers.delete(id);
      host.bufferEpochs.delete(id);
    }
    host.textures.get(targetId)?.destroy();
    host.textures.delete(targetId);
    host.pipelines.delete(pipelineId);
    host.lastRenderTargetId = previousTargetId;
  }
}
