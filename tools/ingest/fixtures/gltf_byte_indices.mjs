/** A self-contained, colored glTF quad with the core UNSIGNED_BYTE index type. */
export function byteIndexedGLB() {
  const binary = new ArrayBuffer(72);
  new Float32Array(binary, 0, 12).set([-.7, -.7, 0, .7, -.7, 0, .7, .7, 0, -.7, .7, 0]);
  new Uint8Array(binary, 48, 16).set([255, 70, 20, 255, 30, 230, 80, 255,
    40, 90, 255, 255, 240, 200, 40, 255]);
  new Uint8Array(binary, 64, 6).set([0, 1, 2, 0, 2, 3]);
  const json = {
    asset: {version: '2.0'}, extensionsUsed: ['KHR_materials_unlit'],
    scene: 0, scenes: [{nodes: [0]}], nodes: [{mesh: 0}],
    meshes: [{primitives: [{attributes: {POSITION: 0, COLOR_0: 1}, indices: 2, material: 0}]}],
    materials: [{doubleSided: true, extensions: {KHR_materials_unlit: {}}}],
    buffers: [{byteLength: binary.byteLength}],
    bufferViews: [{buffer: 0, byteOffset: 0, byteLength: 48},
      {buffer: 0, byteOffset: 48, byteLength: 16}, {buffer: 0, byteOffset: 64, byteLength: 6}],
    accessors: [{bufferView: 0, componentType: 5126, count: 4, type: 'VEC3',
      min: [-.7, -.7, 0], max: [.7, .7, 0]},
    {bufferView: 1, componentType: 5121, count: 4, type: 'VEC4', normalized: true},
    {bufferView: 2, componentType: 5121, count: 6, type: 'SCALAR'}],
  };
  const encoded = new TextEncoder().encode(JSON.stringify(json)), jsonBytes = Math.ceil(encoded.length / 4) * 4;
  const result = new ArrayBuffer(28 + jsonBytes + binary.byteLength), view = new DataView(result), bytes = new Uint8Array(result);
  view.setUint32(0, 0x46546c67, true); view.setUint32(4, 2, true); view.setUint32(8, result.byteLength, true);
  view.setUint32(12, jsonBytes, true); view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(32, 20, 20 + jsonBytes); bytes.set(encoded, 20);
  view.setUint32(20 + jsonBytes, binary.byteLength, true); view.setUint32(24 + jsonBytes, 0x004e4942, true);
  bytes.set(new Uint8Array(binary), 28 + jsonBytes);
  return result;
}
