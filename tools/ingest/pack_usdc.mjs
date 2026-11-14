/** Bounded build-only metadata reads through the exact pinned USDC parser.
 * Geometry stays opaque. The original runtime parser still receives every byte.
 * These instance-local guards do not modify upstream classes or deployed code.
 */
const MAX_ITEMS = 1000000;
const IMAGE_FIELDS = new Set(["default", "connectionPaths", "timeSamples", "info:id"]);
const fail = message => { throw Object.assign(new Error("USD_BUDGET: " + message), {code: "USD_BUDGET"}); };
const unsupported = message => { throw Object.assign(new Error("USD_SOURCE_UNSUPPORTED: " + message), {code: "USD_SOURCE_UNSUPPORTED"}); };

export function parseUsdcImageData(Parser, bytes, maxBytes) {
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const view = new DataView(buffer), parser = new Parser();
  let remaining = maxBytes, operations = 0, depth = 0;
  const charge = amount => {
    if (!Number.isSafeInteger(amount) || amount < 0 || (remaining -= amount) < 0)
      fail("USDC image metadata exceeds its decoded work budget");
  };
  const tick = () => { if (++operations > MAX_ITEMS) fail("USDC image metadata exceeds its operation bound"); };
  const range = (offset, size) => {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size < 0 || offset > buffer.byteLength - size)
      unsupported("USDC metadata range cannot be inspected safely");
  };
  const u64 = offset => {
    range(offset, 8);
    const value = Number(view.getBigUint64(offset, true));
    if (!Number.isSafeInteger(value)) fail("USDC metadata integer exceeds the exact range");
    return value;
  };
  const count = value => {
    if (!Number.isInteger(value) || value < 0 || value > MAX_ITEMS) fail("USDC metadata count exceeds its work bound");
    return value;
  };
  const compressed = view.getUint8(8) !== 0 || view.getUint8(9) >= 4;
  const toc = u64(16), sections = new Map(), sectionCount = count(u64(toc));
  if (sectionCount > 128) fail("USDC section count exceeds its work bound");
  range(toc + 8, sectionCount * 32);
  for (let i = 0; i < sectionCount; i++) {
    const offset = toc + 8 + i * 32;
    const name = new TextDecoder().decode(new Uint8Array(buffer, offset, 16)).split("\0")[0];
    const start = u64(offset + 16), size = u64(offset + 24);
    range(start, size);
    sections.set(name, {start, size}); // The source keeps the last duplicate.
  }
  // Bound every allocation made before values are inspected. The original
  // parser owns LZ4 and integer decompression; no binary decoder is duplicated.
  for (const [name, {start, size}] of sections) {
    if (name === "TOKENS") {
      charge(count(u64(start)) * 16 + u64(start + 8) * 4);
    } else if (name === "STRINGS") {
      charge(count(u64(start)) * 8);
    } else if (name === "FIELDS" || name === "FIELDSETS" || name === "SPECS") {
      const stride = name === "FIELDSETS" ? 4 : name === "SPECS" && view.getUint8(9) === 0 && view.getUint8(10) === 1 ? 16 : 12;
      charge(count(compressed ? u64(start) : Math.floor(size / stride)) * (name === "FIELDS" ? 64 : name === "SPECS" ? 48 : 8));
    } else if (name === "PATHS") {
      charge(count(u64(start)) * 16);
    }
  }
  const wrap = (name, before) => {
    const original = parser[name];
    parser[name] = function (...args) { before.apply(this, args); return original.apply(this, args); };
  };
  const bootstrap = parser._readBootstrap;
  parser._readBootstrap = function () {
    bootstrap.call(this);
    for (const name of ["readUint8", "readInt8", "readUint16", "readInt16", "readUint32", "readInt32",
      "readUint64", "readInt64", "readFloat32", "readFloat64", "readBytes"]) {
      const original = this.reader[name];
      this.reader[name] = function (...args) { tick(); return original.apply(this, args); };
    }
  };
  // The source's compressed path walker recurses into sibling subtrees. Check
  // its exact visitation order first, bounding recursion and expanded strings.
  wrap("_buildPathsFromCompressed", function (indices, tokens, jumps) {
    const pending = [{index: 0, length: 0, depth: 0}];
    while (pending.length) {
      let {index, length, depth} = pending.pop();
      if (depth > 128) fail("USDC path traversal exceeds its depth bound");
      while (index < indices.length) {
        tick();
        const current = index++, jump = jumps[current];
        const token = this.tokens[Math.abs(tokens[current])] || "";
        if (indices[current] < 0 || indices[current] >= this.paths.length)
          unsupported("USDC sparse path indices require the ordinary deployment path");
        const pathLength = length === 0 ? 1 : length + (length === 1 && tokens[current] >= 0 ? 0 : 1) + token.length;
        charge(pathLength * 2);
        if (length === 0) length = 1;
        if (jump > 0) pending.push({index: current + jump, length, depth: depth + 1});
        if (jump > 0 || jump === -1) length = pathLength;
        else if (jump < 0) break;
      }
    }
  });
  wrap("_readPathsRecursive", function (parent, depth = 0) {
    tick();
    if (depth > 128) fail("USDC path traversal exceeds its depth bound");
    const offset = this.reader.tell();
    range(offset, 9);
    if (view.getUint32(offset, true) >= this.paths.length)
      unsupported("USDC sparse path indices require the ordinary deployment path");
    const token = this.tokens[view.getUint32(offset + 4, true)] || "";
    charge((parent.length + token.length + 1) * 2);
  });
  const fieldReps = new Set(), readFields = parser._readFields;
  parser._readFields = function () {
    readFields.call(this);
    for (const field of this.fields ?? []) if (IMAGE_FIELDS.has(this.tokens[field.tokenIndex])) fieldReps.add(field.valueRep);
  };
  wrap("_getFieldsForSpec", function (spec) {
    // Missing field indices do not call _readValue(), so account for those too.
    for (let i = spec.fieldSetIndex, n = 0; i < this.fieldSets.length && n < 10000; i++, n++) {
      tick();
      if (this.fieldSets[i] === -1 || this.fieldSets[i] === 0xffffffff) break;
    }
  });
  const readValue = parser._readValue;
  parser._readValue = function (rep) {
    tick();
    if (depth === 0 && !fieldReps.has(rep)) return undefined;
    const type = rep.typeEnum;
    // Only string-like values, connections and rest-pose samples can name an
    // image. A non-string default must still stop connection/default fallback.
    if (rep.isArray ? type !== 9 && type !== 11 : ![9, 10, 11, 12, 34, 40, 41, 46, 48].includes(type)) return null;
    if (++depth > 128) fail("USDC image values exceed their depth bound");
    try { return readValue.call(this, rep); }
    finally { depth--; }
  };
  wrap("_readArrayValue", function () {
    const offset = this.reader.tell();
    const size = this.version.major === 0 && this.version.minor < 7 ? view.getUint32(offset, true) : u64(offset);
    charge(count(size) * 16);
  });
  wrap("_readScalarValue", function (type) {
    let offset = this.reader.tell();
    if ([40, 41, 48].includes(type)) charge(count(u64(offset)) * 16);
    if (type === 34) {
      const flags = view.getUint8(offset++);
      for (const flag of [2, 4, 32, 64, 8, 16]) if (flags & flag) {
        const size = count(u64(offset));
        charge(size * 16); range(offset + 8, size * 4); offset += 8 + size * 4;
      }
    }
  });
  wrap("_readTimeSamples", function (rep) {
    let offset = rep.payload;
    range(offset, 8);
    offset += Number(view.getBigInt64(offset, true));
    range(offset, 16);
    offset += 8;
    offset += Number(view.getBigInt64(offset, true));
    const size = count(u64(offset));
    charge(size * 32); range(offset + 8, size * 8);
  });
  return parser.parseData(buffer);
}
