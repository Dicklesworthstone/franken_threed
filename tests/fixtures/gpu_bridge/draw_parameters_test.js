import { OPCODE_SET_DRAW_PARAMETERS, WebGpuBridgeHost } from "./bridge_runtime.js";

// Independent WebGPU execution of the agreed range/instance scene. No F3D packet decoder.
async function directReference(device) {
  const texture = device.createTexture({ size: [64, 64], format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 16384,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const vertices = new Float32Array([
    3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0,
    -0.2, -0.5, 0, 0, 0, 0.2, -0.5, 0, 0, 0, 0, 0.5, 0, 0, 0,
  ]);
  const buffer = device.createBuffer({ size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    device.queue.writeBuffer(buffer, 0, vertices);
    const module = device.createShaderModule({ code: `
      struct Out { @builtin(position) position: vec4f, @location(0) color: vec4f };
      @vertex fn vertex(@location(0) p: vec3f, @builtin(instance_index) i: u32) -> Out {
        var out: Out;
        let x = select(0.5, -0.5, i == 5u);
        out.position = vec4f(p.x + x, p.y, p.z, 1.0);
        out.color = select(vec4f(0, 1, 0, 1), vec4f(1, 0, 0, 1), i == 5u);
        return out;
      }
      @fragment fn fragment(in: Out) -> @location(0) vec4f { return in.color; }
    ` });
    const pipeline = device.createRenderPipeline({ layout: "auto",
      vertex: { module, entryPoint: "vertex", buffers: [{ arrayStride: 20,
        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, buffer);
    pass.draw(3, 2, 3, 5);
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: 256 }, [64, 64]);
    device.queue.submit([encoder.finish()]);
    const validation = device.popErrorScope();
    scopeOpen = false;
    const error = await validation;
    if (error) throw new Error(`Direct draw reference: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return pixels;
  } finally {
    if (scopeOpen) await device.popErrorScope();
    texture.destroy(); readback.destroy(); buffer.destroy();
  }
}

function parameterOffset(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const sizes = { 1: 14, 2: 18, 3: 34, 4: 46, 5: 26, 6: 22, 11: 14 };
  let cursor = 16;
  for (let i = 0; i < view.getUint32(8, true); i++) {
    const op = view.getUint16(cursor, true);
    if (op === OPCODE_SET_DRAW_PARAMETERS) return cursor + 2;
    if (!sizes[op]) throw new Error(`Unexpected opcode ${op} in Rust draw-range fixture`);
    cursor += sizes[op];
  }
  throw new Error("Rust packet omitted required draw parameters");
}

function differs(a, b) {
  return a.length !== b.length || a.some((value, i) => value !== b[i]);
}

export async function testDrawParameters(host, buildPacket) {
  if (typeof buildPacket !== "function") throw new Error("Missing required Rust draw-parameters export");
  const packet = buildPacket();
  const offset = parameterOffset(packet);
  const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
  if (view.getUint32(offset, true) !== 2 || view.getUint32(offset + 4, true) !== 3 ||
      view.getUint32(offset + 8, true) !== 5) throw new Error("Rust fixture must request draw(3, 2, 3, 5)");
  const reference = await directReference(host.device);
  await host.executePacket(packet);
  const actual = await host.readbackBuffer(20, 16384);
  if (differs(actual, reference)) throw new Error("Rust draw ranges/instances differ from direct WebGPU pixels");
  for (const [x, rgba] of [[16, [255, 0, 0, 255]], [48, [0, 255, 0, 255]]]) {
    const pixel = actual.subarray(32 * 256 + x * 4, 32 * 256 + x * 4 + 4);
    if (differs(pixel, rgba)) throw new Error(`Instance sample x=${x} has wrong RGBA: ${pixel}`);
  }
  for (const [field, label] of [[0, "zero instances"], [4, "wrong first vertex"], [8, "wrong first instance"]]) {
    const changed = packet.slice();
    new DataView(changed.buffer).setUint32(offset + field, 0, true);
    const negativeHost = new WebGpuBridgeHost();
    try {
      await negativeHost.negotiateAndCreateDevice({ requiredFeatures: [] });
      await negativeHost.executePacket(changed);
      const pixels = await negativeHost.readbackBuffer(20, 16384);
      if (!differs(pixels, reference)) throw new Error(`Planted ${label} was not detected`);
      if (field === 0 && pixels.some((v, i) => v !== (i % 4 === 3 ? 255 : 0))) {
        throw new Error("Zero-instance draw changed the opaque black attachment");
      }
    } finally {
      negativeHost.destroyDevice();
    }
  }
  return "Rust draw(3,2,3,5) matches 16384 direct WebGPU bytes; zero instances and wrong vertex/instance starts differ";
}
