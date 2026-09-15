// Independent direct WebGPU reference; no F3D packet decoding or candidate pixels.
async function directResolve(device, mode) {
  const source = device.createTexture({
    size: [64, 64], format: "rgba8unorm", sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const resolved = device.createTexture({
    size: [64, 64], format: "rgba8unorm", sampleCount: 1,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: 16384, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    let pipeline;
    if (mode === 1) {
      const module = device.createShaderModule({ code: `
        @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
          let p = array<vec2f, 3>(vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(0.0, 0.5));
          return vec4f(p[i], 0.0, 1.0);
        }
        @fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
      ` });
      pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
        multisample: { count: 4 },
      });
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: source.createView(), resolveTarget: resolved.createView(),
      loadOp: "clear", storeOp: "discard",
      clearValue: mode === 0 ? [1, 0, 0, 1] : [0, 0, 0, 1],
    }] });
    if (pipeline) {
      pass.setPipeline(pipeline);
      pass.draw(3);
    }
    pass.end();
    encoder.copyTextureToBuffer(
      { texture: resolved }, { buffer: readback, bytesPerRow: 256 }, [64, 64],
    );
    device.queue.submit([encoder.finish()]);
    const validation = device.popErrorScope();
    scopeOpen = false;
    const error = await validation;
    if (error) throw new Error(`Direct 4x resolve: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return pixels;
  } finally {
    if (scopeOpen) await device.popErrorScope();
    source.destroy();
    resolved.destroy();
    readback.destroy();
  }
}

function assertPixel(pixels, x, y, expected, label) {
  const offset = y * 256 + x * 4;
  const actual = pixels.subarray(offset, offset + 4);
  if (actual.length !== 4 || actual.some((value, i) => value !== expected[i])) {
    throw new Error(`${label}: expected ${expected}, received ${actual}`);
  }
}

function releaseProbe(host, previousTarget) {
  for (const id of [810, 811]) {
    host.textures.get(id)?.destroy();
    host.textures.delete(id);
  }
  host.buffers.get(812)?.destroy();
  host.buffers.delete(812);
  host.bufferEpochs.delete(812);
  host.pipelines.delete(813);
  host.lastRenderTargetId = host.textures.has(previousTarget) ? previousTarget : null;
}

export async function testMultisampleResolve(host, buildPacket) {
  if (typeof buildPacket !== "function") throw new Error("Missing Rust multisample packet export");
  const previousTarget = host.lastRenderTargetId;
  for (const mode of [0, 1]) {
    const reference = await directResolve(host.device, mode);
    try {
      await host.executePacket(buildPacket(mode));
      const source = host.textures.get(810);
      const resolved = host.textures.get(811);
      if (source?.sampleCount !== 4 || resolved?.sampleCount !== 1) {
        throw new Error("Resolve fixture did not create actual 4x and 1x GPU textures");
      }
      const pixels = await host.readbackBuffer(812, 16384);
      if (pixels.length !== reference.length) throw new Error("Resolve readback length differs");
      const mismatch = pixels.findIndex((value, i) => value !== reference[i]);
      if (mismatch !== -1) {
        throw new Error(`Resolve mode ${mode}: byte ${mismatch} is ${pixels[mismatch]}, expected ${reference[mismatch]}`);
      }
      assertPixel(pixels, 32, 32, [255, 0, 0, 255], `Resolve mode ${mode} center`);
      assertPixel(pixels, 0, 0, mode === 0 ? [255, 0, 0, 255] : [0, 0, 0, 255], "Resolve background");
      if (mode === 1 && !pixels.some((value, i) => i % 4 === 0 && value > 0 && value < 255)) {
        throw new Error("4x triangle resolve has no partially covered edge pixels");
      }
    } finally {
      releaseProbe(host, previousTarget);
    }
  }
  // Freeze the six-command mode-1 fixture before changing individual wire fields.
  const packet = buildPacket(1);
  const wire = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
  for (const [offset, opcode] of [[16, 23], [42, 6], [64, 1], [78, 24], [116, 25], [166, 5]]) {
    if (wire.getUint16(offset, true) !== opcode) throw new Error("Unexpected Rust MSAA fixture layout");
  }
  for (const [offset, value, expected] of [
    [38, 2, /unsupported sample count 2/],
    [38, 1, /requires a 4x source/],
    [162, 810, /requires a 4x source/],
    [112, 1, /pipeline sample count does not match/],
  ]) {
    const changed = packet.slice();
    new DataView(changed.buffer, changed.byteOffset, changed.byteLength).setUint32(offset, value, true);
    try {
      const error = await host.executePacket(changed).then(() => null, error => error);
      if (!error || !expected.test(error.message)) {
        throw new Error(`MSAA mutation at ${offset}=${value}: expected ${expected}, received ${error}`);
      }
    } finally {
      releaseProbe(host, previousTarget);
    }
  }
  return "Rust clear-only and triangle packets match all 16384 direct WebGPU bytes each; actual 4x-to-1x textures, source discard, red/black probes, partial edge coverage and four malformed-packet rejections";
}
