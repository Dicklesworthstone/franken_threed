import assert from "node:assert/strict";
import test from "node:test";
import { decoder, fixture, vectors } from "../../tests/fixtures/meshopt/vectors.mjs";
import { createGltfAccessorReader, decodeGltfAnimation } from "./animation_gltf.mjs";
import { decodeGltfAnimationModel } from "./animation_model.mjs";
import { loadGltfAsset } from "./gltf_asset.mjs";

const encode = (model) => new TextEncoder().encode(JSON.stringify(model));

for (const v of vectors.slice(0, 2))
  test(`${v.name}: compressed quantized meshes feed actual geometry, morph and material preflight`, async () => {
    const f = fixture(v);
    f.model.extensionsRequired.push("KHR_mesh_quantization");
    f.model.scenes = [{ nodes: [0] }];
    f.model.nodes = [{ mesh: 0, scale: [0.01, 0.01, 0.01] }];
    f.model.meshes = [
      {
        weights: [0.5],
        primitives: [
          {
            mode: 5,
            attributes: { POSITION: 0, TEXCOORD_0: 1 },
            targets: [{ POSITION: 2 }],
            material: 0,
          },
        ],
      },
    ];
    f.model.accessors = [
      {
        bufferView: 0,
        componentType: 5123,
        count: 4,
        type: "VEC3",
        min: [0, 0, 0],
        max: [300, 300, 0],
      },
      { bufferView: 0, byteOffset: 8, componentType: 5123, count: 4, type: "VEC2" },
      { bufferView: 0, componentType: 5122, count: 4, type: "VEC3" },
    ];
    f.model.materials = [
      {
        extensions: { KHR_materials_unlit: {} },
        pbrMetallicRoughness: {
          baseColorTexture: {
            index: 0,
            extensions: { KHR_texture_transform: { scale: [1 / 500, 1 / 500] } },
          },
        },
      },
    ];
    f.model.textures = [{ source: 0 }];
    f.model.images = [{ uri: "color.png" }];
    const asset = await loadGltfAsset(encode(f.model), {
      baseURL: "https://example.test/scene.gltf",
      meshoptDecoder: decoder,
      fetch: async (url) => {
        assert.ok(url.endsWith("compressed.bin"));
        return new Response(f.encoded);
      },
    });
    const texture = { view: {}, sampler: {} },
      calls = [];
    const result = decodeGltfAnimationModel(asset.json, asset.buffers, {
      resolveTexture: (r) => {
        calls.push(r);
        return texture;
      },
    });
    assert.equal(result.drawables.length, 1);
    const draw = result.drawables[0];
    assert.deepEqual(
      [...draw.geometry.positions],
      [0, 0, 0, 300, 0, 0, 0, 300, 0, 300, 0, 0, 300, 300, 0, 0, 300, 0],
    );
    assert.deepEqual([...draw.geometry.morphTargets[0].positions], [...draw.geometry.positions]);
    assert.equal(draw.geometry.flatNormals, true);
    assert.equal(draw.indices, null);
    assert.deepEqual([...draw.texCoords], [0, 0, 500, 0, 0, 500, 500, 0, 500, 500, 0, 500]);
    assert.deepEqual(draw.uvTransform, [1 / 500, 0, -0, 1 / 500, 0, 0]);
    assert.equal(draw.baseColorTexture.view, texture.view);
    assert.deepEqual(result.definition.nodes[0].scale, [0.01, 0.01, 0.01]);
    assert.deepEqual(result.definition.nodes[0].weights, [0.5]);
    assert.deepEqual(result.source, [{ node: 0, mesh: 0, primitive: 0, material: 0 }]);
    assert.equal(calls.length, 1);
  });

test("meshopt quaternion and exponential filters feed real animation accessor decoding", async () => {
  for (const v of vectors.filter((v) => ["QUATERNION", "EXPONENTIAL"].includes(v.filter))) {
    const f = fixture(v);
    f.model.accessors = [
      {
        bufferView: 0,
        type: "VEC4",
        count: v.count,
        componentType: v.filter === "QUATERNION" ? 5122 : 5126,
        ...(v.filter === "QUATERNION" ? { normalized: true } : {}),
      },
    ];
    const asset = await loadGltfAsset(encode(f.model), {
      baseURL: "https://example.test/scene.gltf",
      meshoptDecoder: decoder,
      fetch: async () => new Response(f.encoded),
    });
    const a = createGltfAccessorReader(asset.json, asset.buffers).read(0);
    const expectedView = new DataView(v.expected.buffer);
    for (let i = 0; i < a.values.length; i++)
      assert.equal(
        a.values[i],
        v.filter === "QUATERNION"
          ? Math.max(expectedView.getInt16(i * 2, true) / 32767, -1)
          : expectedView.getFloat32(i * 4, true),
      );
  }
});

test("compressed keyframe output retains interpolation, original node targets and times", async () => {
  const v = vectors.find((v) => v.filter === "QUATERNION"),
    f = fixture(v);
  const times = new Float32Array([0, 1, 2, 3]);
  f.model.buffers.push({
    byteLength: times.byteLength,
    uri: "data:application/octet-stream;base64," + Buffer.from(times.buffer).toString("base64"),
  });
  f.model.bufferViews.push({ buffer: 2, byteLength: times.byteLength });
  f.model.accessors = [
    { bufferView: 1, type: "SCALAR", count: 4, componentType: 5126, min: [0], max: [3] },
    { bufferView: 0, type: "VEC4", count: 4, componentType: 5122, normalized: true },
  ];
  f.model.nodes = [{}];
  f.model.animations = [
    {
      name: "rotations",
      samplers: [{ input: 0, output: 1, interpolation: "LINEAR" }],
      channels: [{ sampler: 0, target: { node: 0, path: "rotation" } }],
    },
  ];
  const asset = await loadGltfAsset(encode(f.model), {
    baseURL: "https://example.test/scene.gltf",
    meshoptDecoder: decoder,
    fetch: async () => new Response(f.encoded),
  });
  const definition = decodeGltfAnimation(asset.json, asset.buffers),
    channel = definition.clips[0].channels[0];
  assert.equal(channel.node, 0);
  assert.equal(channel.path, "rotation");
  assert.equal(channel.interpolation, "LINEAR");
  assert.equal(channel.quantizedRotation, true);
  assert.deepEqual(channel.times, [0, 1, 2, 3]);
  assert.equal(channel.values.length, 16);
  assert.equal(channel.values[0], 1);
  assert.equal(channel.values[6], 0);
  assert.equal(channel.values[15], 29277 / 32767);
});
