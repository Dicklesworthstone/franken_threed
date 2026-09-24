// Authored fixture: two material alternatives, separate image/UV requirements,
// two scenes sharing a mesh, and one real translation clip. No source assets.
export const VARIANTS = "KHR_materials_variants";
export function materialVariantFixture({ textured = false } = {}) {
  const json = {
    asset: { version: "2.0", copyright: "Synthetic f3d test fixture" },
    scene: 0, scenes: [{ nodes: [0] }, { nodes: [1] }],
    nodes: [{ mesh: 0 }, { mesh: 0, translation: [10, 0, 0] }],
    meshes: [{ primitives: [{ attributes: {}, material: 0, extensions: {
      [VARIANTS]: { mappings: [{ material: 1, variants: [0] }] },
    } }] }],
    extensions: { [VARIANTS]: { variants: [{ name: "blue" }, { name: "plain" }] } },
    extensionsUsed: [VARIANTS, "KHR_materials_clearcoat"], extensionsRequired: [VARIANTS],
    materials: [
      { pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] } },
      { pbrMetallicRoughness: { baseColorFactor: [0, 0, 1, 0.5], metallicFactor: 0.25 },
        alphaMode: "BLEND", doubleSided: true,
        extensions: { KHR_materials_clearcoat: { clearcoatFactor: 0.75 } } },
    ],
    buffers: [], bufferViews: [], accessors: [],
  };
  const buffers = [];
  const attr = (values, type = "VEC3") => {
    const data = new Float32Array(values), buffer = buffers.push(data) - 1;
    json.buffers.push({ byteLength: data.byteLength,
      uri: "data:application/octet-stream;base64," + Buffer.from(data.buffer).toString("base64") });
    const bufferView = json.bufferViews.push({ buffer, byteLength: data.byteLength }) - 1;
    return json.accessors.push({ bufferView, componentType: 5126, type,
      count: values.length / { VEC3: 3, VEC2: 2, SCALAR: 1 }[type] }) - 1;
  };
  const primitive = json.meshes[0].primitives[0];
  primitive.attributes.POSITION = attr([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  Object.assign(json.accessors[0], { min: [0, 0, 0], max: [1, 1, 0] });
  primitive.attributes.NORMAL = attr([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  primitive.attributes.TEXCOORD_0 = attr([0, 0, 1, 0, 0, 1], "VEC2");
  primitive.attributes.TEXCOORD_1 = attr([0.25, 0.5, 0.75, 0.5, 0.25, 1], "VEC2");
  const time = attr([0, 1], "SCALAR"), translation = attr([0, 0, 0, 4, 2, 0]);
  Object.assign(json.accessors[time], { min: [0], max: [1] });
  json.animations = [{ name: "move", samplers: [{ input: time, output: translation }],
    channels: [{ sampler: 0, target: { node: 0, path: "translation" } }] }];
  if (textured) {
    json.extensionsUsed.push("KHR_texture_transform");
    json.images = [{ uri: "red.png" }, { uri: "blue.png" }];
    json.samplers = [{ minFilter: 9729, magFilter: 9729 }];
    json.textures = [{ source: 0, sampler: 0 }, { source: 1, sampler: 0 }];
    json.materials[0].pbrMetallicRoughness.baseColorTexture = { index: 0 };
    json.materials[1].pbrMetallicRoughness.baseColorTexture = { index: 1, texCoord: 1,
      extensions: { KHR_texture_transform: { offset: [0.5, 0.25] } } };
  }
  return { json, buffers, primitive, attr };
}
