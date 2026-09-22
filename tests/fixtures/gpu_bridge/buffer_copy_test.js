// Buffer-to-buffer copy regression using real WebGPU buffers, copyBufferToBuffer, and readback.
// Exercises canonical Rust exports f3d_build_buffer_copy_packet and f3d_build_render_then_copy_packet.

function findCopyCommandPayloadOffset(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const commandEnd = bytes.byteLength - view.getUint32(12, true);
  const opcode = view.getUint16(commandEnd - 42, true);
  if (opcode !== 20) throw new Error(`Expected opcode 20 at ${commandEnd - 42}, found ${opcode}`);
  return commandEnd - 40;
}

async function drainAndCleanup(host, srcId, dstId) {
  try {
    if (host.device?.queue) await host.device.queue.onSubmittedWorkDone();
  } finally {
    for (const id of [srcId, dstId]) {
      if (host.buffers?.has(id)) {
        host.buffers.get(id).destroy();
        host.buffers.delete(id);
      }
      host.bufferEpochs?.delete(id);
    }
  }
}

export async function testBufferCopy(host, wasmExports, canvasContext) {
  const buildPacket =
    wasmExports?.f3d_build_buffer_copy_packet ||
    (typeof wasmExports === "function" ? wasmExports : null);
  if (typeof buildPacket !== "function") {
    throw new Error("Missing required canonical export: f3d_build_buffer_copy_packet");
  }
  const buildRenderCopyPacket = wasmExports?.f3d_build_render_then_copy_packet;
  if (typeof buildRenderCopyPacket !== "function") {
    throw new Error("Missing required canonical export: f3d_build_render_then_copy_packet");
  }
  if (!host || !host.device) throw new Error("WebGpuBridgeHost device not initialized");

  const totalSize = 256;
  const srcOffset = 16;
  const dstOffset = 32;
  const copySize = 64;
  const srcBufferId = 610;
  const dstBufferId = 611;

  // 1. Fill pattern buffer with non-zero bytes so destination zeros are distinct
  const data = new Uint8Array(totalSize);
  for (let i = 0; i < totalSize; i++) data[i] = (i * 7 + 13) & 0xff || 1;

  const assertBufferMatches = (readback, label) => {
    if (readback.byteLength !== totalSize)
      throw new Error(
        `${label} length mismatch: got ${readback.byteLength}, expected ${totalSize}`,
      );
    for (let i = 0; i < copySize; i++) {
      if (readback[dstOffset + i] !== data[srcOffset + i]) {
        throw new Error(
          `${label} mismatch at dst ${dstOffset + i}: got ${readback[dstOffset + i]}, expected ${data[srcOffset + i]}`,
        );
      }
    }
    if (readback.subarray(0, dstOffset).some((b) => b !== 0))
      throw new Error(`${label} prefix not zero`);
    if (readback.subarray(dstOffset + copySize).some((b) => b !== 0))
      throw new Error(`${label} suffix not zero`);
  };

  // 2. Build canonical Rust packet and inject opaque 64-bit epoch with high bit set (0x8000000100000002)
  const packet = buildPacket(data, srcOffset, dstOffset, copySize);
  if (!(packet instanceof Uint8Array) || packet.byteLength === 0) {
    throw new Error("f3d_build_buffer_copy_packet returned empty or invalid packet");
  }
  const copyOffset = findCopyCommandPayloadOffset(packet);
  const testEpoch = 0x8000000100000002n;
  new DataView(packet.buffer, packet.byteOffset).setBigUint64(copyOffset + 32, testEpoch, true);

  // 3. Execute packet, read back destination buffer 611, assert copied bytes, zeros, and epoch
  try {
    await host.executePacket(packet);
    const readback = await host.readbackBuffer(dstBufferId, totalSize);
    assertBufferMatches(readback, "Copy");

    // Opaque epoch with high bit set survives in epochHi/Lo
    if (readback.epochHi >>> 0 !== 0x80000001 || readback.epochLo >>> 0 !== 2) {
      throw new Error(
        `Epoch mismatch: expected hi=0x80000001 lo=2, got hi=${readback.epochHi} lo=${readback.epochLo}`,
      );
    }
  } finally {
    await drainAndCleanup(host, srcBufferId, dstBufferId);
  }

  // 4. Zero-size copy execution: execute packet and read back destination to confirm unchanged zeros
  try {
    const zeroPacket = buildPacket(data, 16, 32, 0);
    await host.executePacket(zeroPacket);
    const zeroReadback = await host.readbackBuffer(dstBufferId, totalSize);
    if (zeroReadback.some((b) => b !== 0))
      throw new Error("Zero-size copy modified destination bytes");
  } finally {
    await drainAndCleanup(host, srcBufferId, dstBufferId);
  }

  // 5. Mutate VALID actual Rust packet and execute on decoder, asserting specific errors
  const assertDecoderRejection = async (badPacket, errPattern, label) => {
    let threw = false;
    try {
      await host.executePacket(badPacket);
    } catch (err) {
      threw = true;
      if (!errPattern.test(err.message)) {
        throw new Error(
          `Decoder rejection mismatch for ${label}: got "${err.message}", expected ${errPattern}`,
        );
      }
    } finally {
      await drainAndCleanup(host, srcBufferId, dstBufferId);
    }
    if (!threw) throw new Error(`Decoder failed to reject: ${label}`);
  };

  const decoderMutations = [
    { off: 4, val: 15n, pat: /source_offset.*multiple of 4/i, lbl: "unaligned source offset" },
    {
      off: 16,
      val: 15n,
      pat: /destination_offset.*multiple of 4/i,
      lbl: "unaligned destination offset",
    },
    { off: 24, val: 15n, pat: /size.*multiple of 4/i, lbl: "unaligned copy size" },
    { off: 4, val: 1024n, pat: /source range out of bounds/i, lbl: "out-of-bounds source range" },
    {
      off: 16,
      val: 1024n,
      pat: /destination range out of bounds/i,
      lbl: "out-of-bounds destination range",
    },
  ];

  for (const { off, val, pat, lbl } of decoderMutations) {
    const mutated = packet.slice();
    new DataView(mutated.buffer, mutated.byteOffset).setBigUint64(copyOffset + off, val, true);
    await assertDecoderRejection(mutated, pat, lbl);
  }

  // Same buffer ID rejection: source and destination must be distinct objects
  const sameBufferPacket = packet.slice();
  new DataView(sameBufferPacket.buffer, sameBufferPacket.byteOffset).setUint32(
    copyOffset + 12,
    610,
    true,
  );
  await assertDecoderRejection(sameBufferPacket, /distinct objects/i, "same buffer id");

  // Shorten only final 40-byte COPY fields while preserving data payload to test truncation guard
  const commandEnd = packet.byteLength - totalSize;
  const truncatedPacket = new Uint8Array(packet.byteLength - 20);
  truncatedPacket.set(packet.subarray(0, commandEnd - 20), 0);
  truncatedPacket.set(packet.subarray(commandEnd), commandEnd - 20);
  await assertDecoderRejection(
    truncatedPacket,
    /Truncated COPY_BUFFER_TO_BUFFER/i,
    "truncated COPY_BUFFER_TO_BUFFER command",
  );

  // 6. Rust packet builder argument rejection
  for (const { src, dst, size, desc } of [
    { src: 200, dst: 32, size: 64, desc: "srcOffset + size > data.len" },
    { src: 16, dst: 200, size: 64, desc: "dstOffset + size > data.len" },
    { src: 16, dst: 32, size: 300, desc: "size > data.len" },
  ]) {
    let rejected = false;
    try {
      buildPacket(data, src, dst, size);
    } catch {
      rejected = true;
    }
    if (!rejected)
      throw new Error(`f3d_build_buffer_copy_packet failed to reject invalid range: ${desc}`);
  }

  // 7. Single-packet pass-before-copy execution using real Rust export f3d_build_render_then_copy_packet
  try {
    const renderCopyPacket = buildRenderCopyPacket(data, srcOffset, dstOffset, copySize);
    if (!(renderCopyPacket instanceof Uint8Array) || renderCopyPacket.byteLength === 0) {
      throw new Error("f3d_build_render_then_copy_packet returned empty or invalid packet");
    }
    await host.executePacket(renderCopyPacket, canvasContext);
    const rcReadback = await host.readbackBuffer(dstBufferId, totalSize);
    assertBufferMatches(rcReadback, "Pass->copy");
  } finally {
    await drainAndCleanup(host, srcBufferId, dstBufferId);
  }

  return `Buffer copy verified: 64 bytes copied from 610[16] to 611[32]; exact data matches, surrounding bytes zero; opaque epoch with high bit set preserved in epochHi/Lo; zero-size copy executed and confirmed unchanged; mutated decoder validation errors asserted; pass->copy single-packet verified on canvas`;
}
