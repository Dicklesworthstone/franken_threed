/** Synthetic KTX2 header/ranges and explicit retained-decoder/WebGPU doubles.
 * No Basis bitstream is encoded/decoded here and no native GPU is executed.
 */
export function syntheticKtx2({
  width = 8,
  height = 8,
  levels = 1 + Math.floor(Math.log2(Math.max(width, height))),
  colorSpace = "srgb",
} = {}) {
  const dfd = 80 + 24 * levels;
  let size = Math.ceil((dfd + 44) / 16) * 16,
    w = width,
    h = height;
  const mips = [];
  for (let i = 0; i < levels; i++) {
    const length = Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    mips.push({ offset: size, length });
    size += length;
    w = Math.max(1, Math.floor(w / 2));
    h = Math.max(1, Math.floor(h / 2));
  }
  const bytes = new Uint8Array(size),
    v = new DataView(bytes.buffer);
  bytes.set([171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10]);
  for (const [offset, value] of [
    [16, 1],
    [20, width],
    [24, height],
    [36, 1],
    [40, levels],
    [48, dfd],
    [52, 44],
    [dfd, 44],
  ])
    v.setUint32(offset, value, true);
  v.setUint16(dfd + 8, 2, true);
  v.setUint16(dfd + 10, 40, true);
  bytes.set(
    [166, colorSpace === "srgb" ? 1 : 0, colorSpace === "srgb" ? 2 : 1, 0, 3, 3, 0, 0, 16],
    dfd + 12,
  );
  bytes[dfd + 30] = 127;
  bytes[dfd + 31] = 3;
  v.setUint32(dfd + 40, 0xffffffff, true);
  mips.forEach((m, i) => {
    v.setBigUint64(80 + i * 24, BigInt(m.offset), true);
    v.setBigUint64(88 + i * 24, BigInt(m.length), true);
    v.setBigUint64(96 + i * 24, BigInt(m.length), true);
    bytes.fill(71 + i, m.offset, m.offset + m.length);
  });
  return bytes;
}
export function retainedDecoderDouble({
  format = 36492,
  block = 4,
  blockBytes = 16,
  onParse,
  onDispose,
} = {}) {
  const calls = [],
    textures = [];
  const decoder = {
    calls,
    textures,
    dispose() {
      throw new Error("Borrowed decoder must not be disposed");
    },
    parse(buffer, onLoad, onError) {
      calls.push(buffer);
      const v = new DataView(buffer),
        width = v.getUint32(20, true),
        height = v.getUint32(24, true),
        count = v.getUint32(40, true),
        dfd = v.getUint32(48, true);
      let w = width,
        h = height;
      const mipmaps = [];
      for (let i = 0; i < count; i++) {
        const data = new Uint8Array(Math.ceil(w / block) * Math.ceil(h / block) * blockBytes);
        data.fill(i + 17);
        mipmaps.push({ width: w, height: h, data });
        w = Math.max(1, Math.floor(w / 2));
        h = Math.max(1, Math.floor(h / 2));
      }
      const texture = {
        format,
        type: 1009,
        flipY: false,
        premultiplyAlpha: false,
        colorSpace: v.getUint8(dfd + 14) === 2 ? "srgb" : "",
        image: { width, height },
        mipmaps,
        disposed: 0,
        dispose() {
          this.disposed++;
          for (const mip of mipmaps) mip.data.fill(0);
          onDispose?.();
        },
      };
      textures.push(texture);
      if (onParse) return onParse({ buffer, texture, onLoad, onError });
      onLoad(texture);
    },
  };
  return decoder;
}
