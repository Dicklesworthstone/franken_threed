// Direct regression for the JS host bridge, using real GPU buffers and mapping.
export async function testReadbackIdReuse(host) {
  const id = 0x7ffffffe;
  if (host.buffers.has(id)) throw new Error("Readback regression ID already in use");
  const usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
  const oldBuffer = host.device.createBuffer({ size: 16, usage });
  const newBuffer = host.device.createBuffer({ size: 16, usage });
  const oldBytes = new Uint8Array(16).fill(37);
  const newBytes = new Uint8Array(16).fill(91);

  try {
    host.device.queue.writeBuffer(oldBuffer, 0, oldBytes);
    host.device.queue.writeBuffer(newBuffer, 0, newBytes);
    host.buffers.set(id, oldBuffer);
    host.bufferEpochs.set(id, { epochHi: 2, epochLo: 7 });

    // Reach the real mapAsync await, then reuse the ID before its continuation.
    const pendingOldReadback = host.readbackBuffer(id, 16);
    host.buffers.set(id, newBuffer);
    host.bufferEpochs.set(id, { epochHi: 3, epochLo: 9 });
    const oldReadback = await pendingOldReadback;
    if (oldReadback.epochHi !== 2 || oldReadback.epochLo !== 7 ||
        !oldReadback.every(byte => byte === 37)) {
      throw new Error(`Old GPU bytes mislabeled after ID reuse: epoch=${oldReadback.epochHi}:${oldReadback.epochLo}`);
    }

    const newReadback = await host.readbackBuffer(id, 16);
    if (newReadback.epochHi !== 3 || newReadback.epochLo !== 9 ||
        !newReadback.every(byte => byte === 91)) {
      throw new Error("Replacement GPU buffer lost its own bytes or epoch");
    }
    if (oldBuffer.mapState !== "unmapped" || newBuffer.mapState !== "unmapped") {
      throw new Error("Readback left a GPU buffer mapped");
    }
    return "JS host bridge: real GPU mapping preserves old and new bytes with their respective epochs across ID reuse";
  } finally {
    host.buffers.delete(id);
    host.bufferEpochs.delete(id);
    oldBuffer.destroy();
    newBuffer.destroy();
  }
}
