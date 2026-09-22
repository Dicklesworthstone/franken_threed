// Golden compressed/decoded bytes from meshoptimizer v1.1 js/meshopt_decoder.test.js,
// blob 6151837440c50ac37934675ea205b0d1d21d3c8d. MIT; see LICENSE.md.
// The retained reference decoder is test-only, never imported by product modules.
export { MeshoptDecoder as decoder } from "./meshopt_decoder_reference.mjs";

const bytes = (head, padding = 0, tail = []) =>
  new Uint8Array([...head, ...Array(padding).fill(0), ...tail]);
const words = (values, size) => {
  const result = new Uint8Array(values.length * size),
    view = new DataView(result.buffer);
  values.forEach((value, i) =>
    size === 2 ? view.setUint16(i * size, value, true) : view.setUint32(i * size, value, true),
  );
  return result;
};
export const vertex = bytes([
  0xa0, 0x01, 0x3f, 0, 0, 0, 0x58, 0x57, 0x58, 0x01, 0x26, 0, 0, 0, 0x01, 0x0c, 0, 0, 0, 0x58, 0x01,
  0x08, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x3f, 0, 0, 0, 0x17, 0x18, 0x17, 0x01, 0x26, 0, 0, 0, 0x01, 0x0c,
  0, 0, 0, 0x17, 0x01, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]);
export const vertexExpected = words(
  [0, 0, 0, 0, 0, 0, 300, 0, 0, 0, 500, 0, 0, 300, 0, 0, 0, 500, 300, 300, 0, 0, 500, 500],
  2,
);
export const vectors = [
  {
    name: "attributes v0",
    mode: "ATTRIBUTES",
    stride: 12,
    count: 4,
    encoded: vertex,
    expected: vertexExpected,
  },
  {
    name: "attributes v1",
    mode: "ATTRIBUTES",
    stride: 12,
    count: 4,
    khr: true,
    encoded: bytes([
      0xa1, 0xee, 0xaa, 0xee, 0, 0x4b, 0x4b, 0x4b, 0, 0, 0x4b, 0, 0, 0x7d, 0x7d, 0x7d, 0, 0, 0x7d,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x62, 0, 0x62,
    ]),
    expected: vertexExpected,
  },
  ...[2, 4].map((stride) => ({
    name: "triangle indices " + stride,
    mode: "TRIANGLES",
    stride,
    count: 15,
    encoded: bytes([
      0xe1, 0xf0, 0x10, 0xfe, 0x1f, 0x3d, 0, 0x0a, 0, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86,
      0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
    ]),
    expected: words([0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 5, 2, 1, 4], stride),
  })),
  ...[2, 4].map((stride) => ({
    name: "index sequence " + stride,
    mode: "INDICES",
    stride,
    count: 6,
    encoded: bytes([0xd1, 0, 4, 0xcd, 1, 4, 7, 0x98, 0x1f, 0, 0, 0, 0]),
    expected: words([0, 1, 51, 2, 49, 1000], stride),
  })),
  {
    name: "octahedral 8",
    mode: "ATTRIBUTES",
    filter: "OCTAHEDRAL",
    stride: 4,
    count: 4,
    encoded: bytes(
      [
        0xa0, 0x01, 0x07, 0, 0, 0, 0x1e, 0x01, 0x3f, 0, 0, 0, 0x8b, 0x8c, 0xfd, 0, 0x01, 0x26, 0, 0,
        0,
      ],
      28,
      [0, 1, 0x7f, 0],
    ),
    expected: bytes([0, 1, 127, 0, 0, 159, 82, 1, 255, 1, 127, 0, 1, 130, 241, 1]),
  },
  {
    name: "octahedral 12",
    mode: "ATTRIBUTES",
    filter: "OCTAHEDRAL",
    stride: 8,
    count: 4,
    encoded: bytes(
      [
        0xa0, 0x01, 0x0f, 0, 0, 0, 0x3d, 0x5a, 0x01, 0x0f, 0, 0, 0, 0x0e, 0x0d, 0x01, 0x3f, 0, 0, 0,
        0x9a, 0x99, 0x26, 0x01, 0x3f, 0, 0, 0, 0x0e, 0x0d, 0x0a, 0, 0, 0x01, 0x26, 0, 0, 0,
      ],
      25,
      [0, 0, 1, 0, 0xff, 7, 0, 0],
    ),
    expected: words(
      [0, 16, 32767, 0, 0, 32621, 3088, 1, 32764, 16, 471, 0, 307, 28541, 16093, 1],
      2,
    ),
  },
  {
    name: "quaternion 12",
    mode: "ATTRIBUTES",
    filter: "QUATERNION",
    stride: 8,
    count: 4,
    encoded: bytes(
      [
        0xa0, 0x01, 0x0f, 0, 0, 0, 0x3d, 0x5a, 0x01, 0x0f, 0, 0, 0, 0x0e, 0x0d, 0x01, 0x3f, 0, 0, 0,
        0x9a, 0x99, 0x26, 0x01, 0x3f, 0, 0, 0, 0x0e, 0x0d, 0x0a, 0, 0, 0x01, 0x2a, 0, 0, 0,
      ],
      25,
      [0, 0, 1, 0, 0, 0, 0xfc, 7],
    ),
    expected: words(
      [32767, 0, 11, 0, 0, 25013, 0, 21166, 11, 0, 23504, 22830, 158, 14715, 0, 29277],
      2,
    ),
  },
  {
    name: "exponential",
    mode: "ATTRIBUTES",
    filter: "EXPONENTIAL",
    stride: 16,
    count: 1,
    encoded: bytes(
      [0xa0],
      32,
      [0, 0, 0, 0, 3, 0, 0, 0xff, 0xf7, 0xff, 0xff, 2, 0xff, 0xff, 0x7f, 0xfe],
    ),
    expected: words([0, 0x3fc00000, 0xc2100000, 0x49fffffe], 4),
  },
  {
    name: "color 8",
    mode: "ATTRIBUTES",
    filter: "COLOR",
    stride: 4,
    count: 4,
    khr: true,
    encoded: bytes(
      [
        0xa0, 0x01, 0x3f, 0, 0, 0, 0x7e, 0x7d, 0x4c, 0x01, 0x3f, 0, 0, 0, 0xfd, 0xfd, 0xfe, 0x01,
        0x3f, 0, 0, 0, 0x83, 0x82, 0x80, 0x01, 0x3f, 0, 0, 0, 0x7d, 0x3f, 0x7e,
      ],
      28,
      [0x40, 0x7f, 0xc1, 0xff],
    ),
    expected: bytes([254, 1, 0, 255, 0, 254, 0, 128, 1, 0, 255, 64, 102, 102, 102, 191]),
  },
  {
    name: "color 12",
    mode: "ATTRIBUTES",
    filter: "COLOR",
    stride: 8,
    count: 4,
    khr: true,
    encoded: bytes(
      [
        0xa0, 0x01, 0x1b, 0, 0, 0, 0xcc, 0x01, 0x3f, 0, 0, 0, 0x06, 0x05, 0x04, 0x01, 0x29, 0, 0, 0,
        0x01, 0x3f, 0, 0, 0, 0x0d, 0x0f, 0x10, 0x01, 0x38, 0, 0, 0, 0x03, 0x01, 0x3f, 0, 0, 0, 0x16,
        0x15, 0x08, 0x01, 0x21, 0, 0, 0, 0x01, 0x3f, 0, 0, 0, 0x05, 0x03, 0x06,
      ],
      24,
      [0, 4, 0xff, 7, 1, 0xfc, 0xff, 0x0f],
    ),
    expected: words(
      [65519, 16, 0, 65535, 0, 65519, 0, 32776, 16, 0, 65535, 16388, 26214, 26214, 26214, 49147],
      2,
    ),
  },
];
export function fixture(
  v = vectors[0],
  { required = true, offset = 4, tagged = true, fallback = false } = {},
) {
  const ext = v.khr ? "KHR_meshopt_compression" : "EXT_meshopt_compression",
    size = v.count * v.stride;
  const encoded = bytes(Array(offset).fill(0), 0, [...v.encoded, 97, 98, 99]);
  const model = {
    asset: { version: "2.0" },
    extensionsUsed: [ext],
    extensionsRequired: required ? [ext] : [],
    buffers: [
      { byteLength: encoded.length, uri: "compressed.bin" },
      {
        byteLength: size,
        ...(fallback ? { uri: "fallback.bin" } : {}),
        ...(tagged ? { extensions: { [ext]: { fallback: true } } } : {}),
      },
    ],
    bufferViews: [
      {
        buffer: 1,
        byteLength: size,
        ...(v.mode === "ATTRIBUTES" ? { byteStride: v.stride } : {}),
        extensions: {
          [ext]: {
            buffer: 0,
            byteOffset: offset,
            byteLength: v.encoded.length,
            byteStride: v.stride,
            count: v.count,
            mode: v.mode,
            ...(v.filter ? { filter: v.filter } : {}),
          },
        },
      },
    ],
  };
  return { model, buffers: [encoded, fallback ? v.expected.slice() : null], ext, encoded, v };
}
