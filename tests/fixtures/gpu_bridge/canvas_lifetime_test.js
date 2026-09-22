import {
  OPCODE_RENDER_PASS,
  PACKET_MAGIC,
  PACKET_VERSION,
  TARGET_CANVAS,
} from "./bridge_runtime.js";

// Hand-authored input to the production JS decoder, not evidence of Rust lowering.
function canvasPass(pipelineId, load, clearColor) {
  const bytes = new Uint8Array(16 + 46);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, PACKET_MAGIC, true);
  view.setUint16(4, PACKET_VERSION, true);
  view.setUint32(8, 1, true);
  view.setUint16(16, OPCODE_RENDER_PASS, true);
  view.setUint32(18, TARGET_CANVAS | (load << 8) | (1 << 24), true);
  view.setUint32(22, 1, true);
  clearColor.forEach((value, i) => view.setFloat32(26 + i * 4, value, true));
  view.setUint32(42, pipelineId, true);
  view.setUint32(50, pipelineId ? 3 : 0, true);
  return bytes;
}

export async function testCanvasSubmissionLifetime(host) {
  const device = host.device;
  const pipelineId = 0x7fffff10;
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 1;
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Canvas lifetime test requires a real GPUCanvasContext");
  const format = navigator.gpu.getPreferredCanvasFormat();
  const readback = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const clear = canvasPass(0, 0, [1, 0, 0, 1]);

  try {
    await host.withErrorScopes(["validation"], () => {
      const module = device.createShaderModule({
        code: `
        @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
          let p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
          return vec4(p[i], 0., 1.);
        }
        @fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
          if (p.x < 1.) { discard; }
          return vec4(0., 0., 1., 1.);
        }
      `,
      });
      const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
      host.pipelines.set(pipelineId, { pipeline, hasUniformBuffer: false });
    });

    async function render(load) {
      context.configure({
        device,
        format,
        alphaMode: "opaque",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      // All acquisition, encoding, submissions and the copy happen in this task.
      // queue.submit does not expire the texture; awaiting presentation could.
      const texture = context.getCurrentTexture();
      const first = host.executePacket(clear, context);
      const second = host.executePacket(canvasPass(pipelineId, load, [0, 0, 0, 1]), context);
      const sameTexture = texture === context.getCurrentTexture();
      const copied = host.withErrorScopes(["validation"], () => {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: 256 }, [2, 1, 1]);
        device.queue.submit([encoder.finish()]);
      });
      await Promise.all([first, second, copied]);
      if (!sameTexture)
        throw new Error("Canvas texture changed within one synchronous rendering interval");
      await readback.mapAsync(GPUMapMode.READ);
      let pixels;
      try {
        pixels = new Uint8Array(readback.getMappedRange()).slice(0, 8);
      } finally {
        readback.unmap();
      }
      if (format === "bgra8unorm") {
        for (const i of [0, 4]) [pixels[i], pixels[i + 2]] = [pixels[i + 2], pixels[i]];
      }
      return Array.from(pixels);
    }

    const expected = [255, 0, 0, 255, 0, 0, 255, 255];
    const good = await render(1);
    if (good.some((v, i) => v !== expected[i])) {
      throw new Error(`Canvas submissions lost prefix or resumed draw: ${good}`);
    }
    // Change the actual GPU load operation: the prefix must now be erased.
    const broken = await render(0);
    const erased = [0, 0, 0, 255, 0, 0, 255, 255];
    if (broken.some((v, i) => v !== erased[i])) {
      throw new Error(
        `Canvas clear mutation did not produce the expected failing image: ${broken}`,
      );
    }

    let missingRejected = false;
    try {
      await host.executePacket(clear);
    } catch (error) {
      missingRejected = /canvas/i.test(error.message);
    }
    if (!missingRejected)
      throw new Error("Canvas packet silently succeeded without a canvas context");
    return "JS host bridge: two submissions preserve red/blue canvas pixels; real clear mutation erases red; missing canvas rejected";
  } finally {
    host.pipelines.delete(pipelineId);
    context.unconfigure();
    readback.destroy();
  }
}
