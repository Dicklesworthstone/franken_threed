import {
  PACKET_MAGIC, PACKET_VERSION, OPCODE_RENDER_PASS,
  OPCODE_COPY_TEXTURE_TO_BUFFER, TARGET_OFFSCREEN,
} from "./bridge_runtime.js";

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
      const module = device.createShaderModule({ code: `
        @group(0) @binding(0) var<uniform> color: vec4<f32>;
        @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
          let p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
          return vec4(p[i], 0., 1.);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> { return color; }
      ` });
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
    return "JS host bridge: pipeline switch from 16-byte to 32-byte layout rebinds the same uniform buffer and renders green";
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
