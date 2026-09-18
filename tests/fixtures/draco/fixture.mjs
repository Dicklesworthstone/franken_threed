// Recording codec boundary, NOT a Draco bitstream decoder. Attribute/index data
// are explicit fixture outputs; compressed input is an opaque transport token.
export const EXT = 'KHR_draco_mesh_compression';
export function fixture({componentType = 5126, normalized = false, offset = 4, indices = true} = {}) {
  const Types = {5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array};
  const bytes = new Uint8Array([70,70,70,70,68,82,65,67,79,1,2,3,70,70,70,70]);
  const json = {asset: {version: '2.0'}, extensionsUsed: [EXT], extensionsRequired: [EXT],
    buffers: [{byteLength: bytes.length}], bufferViews: [{buffer: 0, byteOffset: offset, byteLength: 8}],
    accessors: [{type:'VEC3', componentType, count:3, normalized, min:[0,0,0], max:[10,20,0]},
      {type:'SCALAR',componentType:5123,count:3}],
    meshes: [{primitives: [{attributes: {POSITION: 0}, ...(indices ? {indices: 1} : {}), extensions: {[EXT]: {bufferView: 0, attributes: {POSITION: 7}}}}]}],
    nodes: [{mesh: 0}], scenes: [{nodes: [0]}]};
  const calls = [], geometries = [];
  const decoder = {decodeGeometry(buffer, config) {
    calls.push({buffer, config});
    const geometry = {attributes: {}, index: {itemSize:1,count:3,array:new Uint32Array([0,1,2])}, disposed:0,
      dispose() { this.disposed++; }};
    for (const key of Object.keys(config.attributeIDs)) geometry.attributes[key] = {
      itemSize:3,count:3,array:new Types[componentType]([0,0,0,10,0,0,0,20,0]),normalized:false,
    };
    geometries.push(geometry); return Promise.resolve(geometry);
  }};
  return {json, buffers: [bytes], bytes, calls, geometries, decoder, primitive: json.meshes[0].primitives[0], Type: Types[componentType]};
}
