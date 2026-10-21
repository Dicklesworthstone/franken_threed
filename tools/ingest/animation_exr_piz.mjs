/** OpenEXR PIZ block decoding: canonical Huffman, inverse wavelets and range LUT.
 * Format/algorithm references: OpenEXR v3.3.2 internal_piz.c and v3.1.5 ImfHuf.cpp.
 * No I/O, native binding, image allocation, third-party runtime or source mutation.
 * The owning EXR parser validates channels/dimensions and charges scratch first.
 * This implementation uses an exact BigInt canonical table (up to 58-bit codes),
 * a 14-bit fast lookup, and bounded long-code fallback; no imprecise JS bit shifts.
 */
const SYMBOLS = 65537, FAST_BITS = 14, FAST_SIZE = 1 << FAST_BITS;
// Upper bound on typed table storage: lengths, ordered symbols, fast entries,
// counts/offsets/cursors and the range LUT. BigInt/JS metadata is not GPU storage.
export const EXR_PIZ_TABLE_BYTES = SYMBOLS * 5 + FAST_SIZE * 4 + 59 * 12 + 65536 * 2;

function bits(bytes, start, length, fail) {
  let position = 0;
  return {
    get position() { return position; },
    get remaining() { return length - position; },
    read(count) {
      if (count > length - position) fail('DATA', 'Truncated PIZ Huffman bits');
      let result = 0;
      while (count > 0) {
        const byte = start + Math.floor(position / 8), used = position % 8;
        const take = Math.min(count, 8 - used);
        result = result * 2 ** take + ((bytes[byte] >>> (8 - used - take)) & ((1 << take) - 1));
        count -= take; position += take;
      }
      return result;
    },
    peek(count) { const saved = position, result = this.read(count); position = saved; return result; },
  };
}

function huffman(input, output, fail) {
  if (input.length < 20) fail('DATA', 'Truncated PIZ Huffman header');
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
  const minimum = view.getUint32(0, true), maximum = view.getUint32(4, true);
  const tableBytes = view.getUint32(8, true), bitCount = view.getUint32(12, true);
  if (minimum >= maximum || maximum >= SYMBOLS || !tableBytes || !bitCount || view.getUint32(16, true) !== 0 ||
      20 + tableBytes + Math.ceil(bitCount / 8) !== input.length) fail('DATA', 'Invalid PIZ Huffman extent or symbol range');
  const lengths = new Uint8Array(maximum + 1), counts = new Uint32Array(59);
  const table = bits(input, 20, tableBytes * 8, fail);
  for (let symbol = minimum; symbol <= maximum;) {
    const length = table.read(6);
    if (length >= 59) {
      const run = length === 63 ? table.read(8) + 6 : length - 57;
      if (run > maximum + 1 - symbol) fail('DATA', 'PIZ zero-code run exceeds table');
      symbol += run;
    } else { lengths[symbol++] = length; if (length) counts[length]++; }
  }
  if (Math.ceil(table.position / 8) !== tableBytes || !lengths[maximum]) fail('DATA', 'Invalid PIZ code table or run marker');
  const first = Array(59).fill(0n), offsets = new Uint32Array(59), cursor = new Uint32Array(59);
  let code = 0n, previousEnd = 0n, symbolCount = 0;
  // OpenEXR's canonical order is reversed: long codes are numerically smaller.
  // Check intervals at a common exact precision before filling the fast table.
  // Oversubscribed or prefix-colliding tables cannot cause unchecked expansion.
  for (let length = 58; length >= 1; length--) {
    first[length] = code;
    const end = code + BigInt(counts[length]), shift = BigInt(58 - length);
    if (counts[length]) {
      if ((code << shift) < previousEnd || end > (1n << BigInt(length))) fail('DATA', 'Overlapping PIZ Huffman codes');
      previousEnd = end << shift;
    }
    code = end >> 1n;
    offsets[length] = symbolCount; cursor[length] = symbolCount; symbolCount += counts[length];
  }
  const symbols = new Uint32Array(symbolCount), fast = new Uint32Array(FAST_SIZE);
  for (let symbol = minimum; symbol <= maximum; symbol++) {
    const length = lengths[symbol]; if (!length) continue;
    const index = cursor[length]++, value = first[length] + BigInt(index - offsets[length]);
    symbols[index] = symbol;
    if (length <= FAST_BITS) {
      const begin = Number(value) * 2 ** (FAST_BITS - length), size = 2 ** (FAST_BITS - length);
      fast.fill(symbol * 64 + length, begin, begin + size);
    }
  }
  const data = bits(input, 20 + tableBytes, bitCount, fail); let out = 0;
  while (data.remaining) {
    let symbol = -1;
    if (data.remaining >= FAST_BITS) {
      const entry = fast[data.peek(FAST_BITS)];
      if (entry) { data.read(entry & 63); symbol = entry >>> 6; }
    }
    if (symbol < 0) {
      let value = 0n;
      for (let length = 1; length <= 58; length++) {
        value = (value << 1n) | BigInt(data.read(1));
        const offset = value - first[length];
        if (offset >= 0n && offset < BigInt(counts[length])) { symbol = symbols[offsets[length] + Number(offset)]; break; }
      }
      if (symbol < 0) fail('DATA', 'Invalid PIZ Huffman code');
    }
    if (symbol === maximum) {
      const run = data.read(8);
      if (!out || run > output.length - out) fail('DATA', 'Invalid PIZ repeat extent');
      output.fill(output[out - 1], out, out + run); out += run;
    } else {
      if (out === output.length) fail('DATA', 'PIZ output exceeds declared block');
      output[out++] = symbol;
    }
  }
  if (out !== output.length) fail('DATA', 'Incomplete PIZ Huffman output');
}

function inverseWavelet(words, base, width, height, step, maximum) {
  const stride = width * step, small = maximum < 16384;
  function pair(a, b) {
    const low = words[a], high = words[b];
    if (small) {
      const l = (low << 16) >> 16, h = (high << 16) >> 16;
      const first = l + (h & 1) + (h >> 1);
      words[a] = first; words[b] = first - h;
    } else {
      const second = (low - (high >> 1)) & 65535;
      words[a] = high + second - 32768; words[b] = second;
    }
  }
  let span = 1;
  while (span * 2 <= Math.min(width, height)) span *= 2;
  for (let distance = span / 2; distance >= 1; span = distance, distance /= 2) {
    let y = 0;
    for (; y <= height - span; y += span) {
      let x = 0;
      for (; x <= width - span; x += span) {
        const a = base + y * stride + x * step, b = a + distance * step;
        const c = a + distance * stride, d = c + distance * step;
        pair(a, c); pair(b, d); pair(a, b); pair(c, d);
      }
      if (width & distance) { const a = base + y * stride + x * step; pair(a, a + distance * stride); }
    }
    if (height & distance) for (let x = 0; x <= width - span; x += span) {
      const a = base + y * stride + x * step; pair(a, a + distance * step);
    }
  }
}

/** Decode a compressed block into scanline-major, channel-planar little-endian
 * bytes. FLOAT/UINT are wavelet-coded as independent 16-bit words, not numbers.
 * Raw-fallback blocks are handled by the owner before reaching this function.
 */
export function decodeExrPiz(input, width, height, channels, fail) {
  if (input.length < 8) fail('DATA', 'Truncated PIZ range header');
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
  const minimum = view.getUint16(0, true), maximum = view.getUint16(2, true);
  if (maximum >= 8192) fail('DATA', 'PIZ range bitmap exceeds 16-bit alphabet');
  const bitmapBytes = minimum <= maximum ? maximum - minimum + 1 : 0;
  const sizeAt = 4 + bitmapBytes;
  if (sizeAt + 4 > input.length || view.getUint32(sizeAt, true) !== input.length - sizeAt - 4) fail('DATA', 'Invalid PIZ Huffman byte extent');
  const lut = new Uint16Array(65536); let used = 1; // zero is always implicit.
  for (let value = 1; value < 65536; value++) {
    const byte = value >>> 3;
    if (byte >= minimum && byte <= maximum && (input[4 + byte - minimum] & (1 << (value & 7)))) lut[used++] = value;
  }
  const rowBytes = width * channels.reduce((sum, channel) => sum + channel.bytes, 0);
  const rawBytes = rowBytes * height, words = new Uint16Array(rawBytes / 2);
  huffman(input.subarray(sizeAt + 4), words, fail);
  let base = 0;
  for (const channel of channels) {
    const step = channel.bytes / 2;
    for (let part = 0; part < step; part++) inverseWavelet(words, base + part, width, height, step, used - 1);
    base += width * height * step;
  }
  // Validate the inverse range before lookup; undefined typed-array values
  // must not silently turn malformed compressed data into black pixels.
  for (let i = 0; i < words.length; i++) {
    if (words[i] >= used) fail('DATA', 'PIZ wavelet value outside range LUT');
    words[i] = lut[words[i]];
  }
  const result = new Uint8Array(rawBytes), output = new DataView(result.buffer); let at = 0;
  for (let y = 0; y < height; y++) {
    base = 0;
    for (const channel of channels) {
      const count = width * channel.bytes / 2, start = base + y * count;
      for (let x = 0; x < count; x++, at += 2) output.setUint16(at, words[start + x], true);
      base += height * count;
    }
  }
  return result;
}
