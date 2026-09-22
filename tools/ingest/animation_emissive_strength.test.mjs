import assert from "node:assert/strict";
import test from "node:test";
import { decodeGltfAnimationModel, prepareGltfAnimationModel } from "./animation_model.mjs";
import { createAnimationModelExporter } from "./animation_model_export.mjs";
import { exportAnimationPoseGLB } from "./animation_pose_export.mjs";
import { imageBytes, pngBase64 } from "./fixtures/animation/image_fixture.mjs";

// Primary contract: linear emission = core factor * emissiveStrength; textures
// remain sRGB and alpha is unaffected. This suite executes model decoding and
// binary serialization, not a GPU shader, bloom, illumination or animation.
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_emissive_strength
const EXT = "KHR_materials_emissive_strength";
const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const texture = () => ({ view: {}, sampler: {} });
const encoded = () => ({ bytes: imageBytes(pngBase64), mimeType: "image/png" });
function fixture(strength = 4) {
  const model = {
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: {}, material: 0 }] }],
      materials: [
        {
          emissiveFactor: [0.25, 0.5, 1],
          pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 0.75, 0.3] },
          alphaMode: "BLEND",
          extensions: { [EXT]: { emissiveStrength: strength } },
        },
      ],
      buffers: [],
      bufferViews: [],
      accessors: [],
    },
    buffers = [];
  const attributes = model.meshes[0].primitives[0].attributes;
  for (const [name, type, values] of [
    ["POSITION", "VEC3", [0, 0, 0, 1, 0, 0, 0, 1, 0]],
    ["NORMAL", "VEC3", [0, 0, 1, 0, 0, 1, 0, 0, 1]],
    ["TEXCOORD_0", "VEC2", [0, 0, 1, 0, 0, 1]],
  ]) {
    const data = new Float32Array(values),
      buffer = buffers.push(data) - 1;
    model.buffers.push({ byteLength: data.byteLength });
    const bufferView = model.bufferViews.push({ buffer, byteLength: data.byteLength }) - 1;
    attributes[name] =
      model.accessors.push({
        bufferView,
        componentType: 5126,
        type,
        count: 3,
        ...(name === "POSITION" ? { min: [0, 0, 0], max: [1, 1, 0] } : {}),
      }) - 1;
  }
  return { model, buffers, material: model.materials[0] };
}
const decode = (f) => decodeGltfAnimationModel(f.model, f.buffers, { resolveTexture: texture });
function maps(f) {
  f.model.textures = [{ source: 0 }];
  f.model.images = [{ uri: "material.png" }];
  f.material.emissiveTexture = { index: 0 };
  f.material.occlusionTexture = { index: 0, strength: 0.25 };
  return f;
}
function frame(decoded) {
  const pose = { version: 1, disposed: false, nodeCount: decoded.definition.nodes.length };
  const entries = decoded.drawables.map((drawable, i) => ({
    drawable,
    source: decoded.source[i],
    deformer: {
      vertexCount: drawable.geometry.positions.length / 3,
      positions: drawable.geometry.positions,
      normals: drawable.geometry.normals,
      worldMatrix: new Float64Array(I),
      poseVersion: 1,
      disposed: false,
    },
  }));
  return { pose, entries };
}
function unpack(buffer) {
  const view = new DataView(buffer);
  assert.equal(view.getUint32(0, true), 0x46546c67);
  const size = view.getUint32(12, true);
  assert.equal(view.getUint32(8, true), buffer.byteLength);
  return {
    json: JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, size))),
    bin: new Uint8Array(buffer, 28 + size),
  };
}
async function roundtrip(decoded, options = {}) {
  const f = frame(decoded),
    r = unpack(
      await exportAnimationPoseGLB(f.pose, f.entries, { resolveTexture: encoded, ...options }),
    );
  return { ...r, reloaded: decodeGltfAnimationModel(r.json, [r.bin], { resolveTexture: texture }) };
}
const close = (a, b) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) =>
    assert.ok(Math.abs(v - b[i]) <= Math.max(1, Math.abs(b[i])) * 1e-12, `${v} != ${b[i]}`),
  );
};

for (const strength of [0, 0.5, 1, 4, 16])
  test(`emissive strength ${strength} scales linear RGB without touching base color or alpha`, () => {
    const f = fixture(strength),
      before = structuredClone(f.model),
      d = decode(f).drawables[0];
    assert.deepEqual(d.emissiveFactor, [0.25 * strength, 0.5 * strength, strength]);
    assert.deepEqual(d.baseColor, [0.5, 0.25, 0.75, 0.3]);
    assert.equal(d.alphaMode, "BLEND");
    assert.deepEqual(f.model, before);
  });

test("empty extension defaults to one and missing core emission remains black", () => {
  const f = fixture();
  f.material.extensions[EXT] = { extras: { note: "metadata" } };
  assert.deepEqual(decode(f).drawables[0].emissiveFactor, [0.25, 0.5, 1]);
  delete f.material.emissiveFactor;
  f.material.extensions[EXT].emissiveStrength = 100;
  assert.deepEqual(decode(f).drawables[0].emissiveFactor, [0, 0, 0]);
});

test("a required emissive-strength extension is supported, while unknown required extensions still fail", () => {
  const f = fixture();
  f.model.extensionsRequired = [EXT];
  f.model.extensionsUsed = [EXT];
  assert.deepEqual(decode(f).drawables[0].emissiveFactor, [1, 2, 4]);
  f.model.extensionsRequired.push("EXT_unknown");
  assert.throws(() => decode(f), { code: "GLTF_MODEL_UNSUPPORTED" });
});

test("HDR emissive maps keep sRGB resolution distinct from a shared linear occlusion map", async () => {
  const f = maps(fixture()),
    prepared = prepareGltfAnimationModel(f.model, f.buffers);
  assert.deepEqual(
    prepared.textureRequests.map((r) => r.colorSpace),
    ["srgb", "linear"],
  );
  const d = prepared.resolveTextures(texture),
    uses = [];
  const r = await roundtrip(d, {
    resolveTexture: (_, { colorSpaces }) => {
      uses.push(colorSpaces);
      return encoded();
    },
  });
  assert.deepEqual(uses, [["srgb"], ["linear"]]);
  assert.deepEqual(r.json.materials[0].emissiveFactor, [0.25, 0.5, 1]);
  assert.equal(r.json.materials[0].extensions[EXT].emissiveStrength, 4);
  assert.equal(r.json.materials[0].occlusionTexture.strength, 0.25);
  assert.deepEqual(r.reloaded.drawables[0].emissiveFactor, [1, 2, 4]);
  assert.notEqual(
    r.reloaded.drawables[0].emissiveTexture,
    r.reloaded.drawables[0].occlusionTexture,
  );
});

test("prepared emission is detached from JSON before asynchronous texture work", () => {
  const f = maps(fixture()),
    prepared = prepareGltfAnimationModel(f.model, f.buffers);
  f.material.extensions[EXT].emissiveStrength = 999;
  f.material.emissiveFactor.fill(0);
  assert.deepEqual(prepared.resolveTextures(texture).drawables[0].emissiveFactor, [1, 2, 4]);
});

for (const strength of [-1, NaN, Infinity, -Infinity, null, "2"])
  test(`invalid emissive strength ${String(strength)} fails before texture resolution`, () => {
    const f = maps(fixture(strength));
    let calls = 0;
    assert.throws(
      () =>
        decodeGltfAnimationModel(f.model, f.buffers, {
          resolveTexture: () => {
            calls++;
            return texture();
          },
        }),
      { code: "GLTF_MODEL_VALUE" },
    );
    assert.equal(calls, 0);
  });

test("invalid extension objects, unknown fields and nested extensions cannot be silently discarded", () => {
  for (const extension of [
    null,
    [],
    1,
    { emissiveStrength: 2, texture: 0 },
    { extensions: { EXT_unknown: {} } },
  ]) {
    const f = maps(fixture());
    f.material.extensions[EXT] = extension;
    let calls = 0;
    assert.throws(() =>
      decodeGltfAnimationModel(f.model, f.buffers, {
        resolveTexture: () => {
          calls++;
          return texture();
        },
      }),
    );
    assert.equal(calls, 0);
  }
});

test("core emissive factors remain bounded, and scaling overflow is rejected before texture I/O", () => {
  for (const [rgb, strength] of [
    [[2, 0, 0], 1],
    [[-1, 0, 0], 1],
    [[1, 1, 1], 1e100],
  ]) {
    const f = maps(fixture(strength));
    f.material.emissiveFactor = rgb;
    let calls = 0;
    assert.throws(
      () =>
        decodeGltfAnimationModel(f.model, f.buffers, {
          resolveTexture: () => {
            calls++;
            return texture();
          },
        }),
      { code: "GLTF_MODEL_VALUE" },
    );
    assert.equal(calls, 0);
  }
  // The scalar itself need not fit a GPU word: only the folded emission is uploaded.
  const f = fixture(1e200);
  f.material.emissiveFactor = [1e-200, 0, 0];
  close(decode(f).drawables[0].emissiveFactor, [1, 0, 0]);
});

test("KHR_materials_unlit and emissive strength are mutually exclusive on one material", () => {
  const f = fixture();
  f.material.extensions.KHR_materials_unlit = {};
  assert.throws(() => decode(f), { code: "GLTF_MODEL_MATERIAL" });
});

for (const rgb of [
  [0, 0, 0],
  [0.1, 0.2, 0.3],
  [1, 0.5, 0],
  [2, 0.5, 4],
  [0, 1000, 0],
  [0.01, 0.2, 17],
])
  test(`runtime emission ${rgb.join("/")} exports and reloads without clipping`, async () => {
    const decoded = decode(fixture());
    decoded.drawables[0].emissiveFactor = rgb;
    const r = await roundtrip(decoded),
      m = r.json.materials[0],
      strength = Math.max(...rgb);
    assert.ok(m.emissiveFactor.every((n) => n >= 0 && n <= 1));
    close(r.reloaded.drawables[0].emissiveFactor, rgb);
    if (strength > 1) {
      assert.equal(m.extensions[EXT].emissiveStrength, strength);
      assert.deepEqual(r.json.extensionsUsed, [EXT]);
      assert.deepEqual(r.json.extensionsRequired, [EXT]);
    } else {
      assert.equal(m.extensions, undefined);
      assert.equal(r.json.extensionsUsed, undefined);
    }
  });

test("mixed HDR and unlit materials retain both required declarations in any material order", async () => {
  for (const reverse of [false, true]) {
    const f = fixture();
    f.model.materials.push({ extensions: { KHR_materials_unlit: {} } });
    f.model.meshes[0].primitives.push({ ...f.model.meshes[0].primitives[0], material: 1 });
    if (reverse) f.model.meshes[0].primitives.reverse();
    const r = await roundtrip(decode(f));
    assert.deepEqual([...r.json.extensionsRequired].sort(), [EXT, "KHR_materials_unlit"].sort());
    assert.deepEqual([...r.json.extensionsUsed].sort(), [EXT, "KHR_materials_unlit"].sort());
    assert.equal(r.reloaded.drawables.find((d) => d.shading === "unlit").emissiveFactor, undefined);
    assert.deepEqual(
      r.reloaded.drawables.find((d) => d.shading === "metallic-roughness").emissiveFactor,
      [1, 2, 4],
    );
  }
});

test("multiple HDR materials deduplicate declarations and preserve their own intensities", async () => {
  const f = fixture();
  f.model.materials.push({
    emissiveFactor: [0, 1, 0],
    extensions: { [EXT]: { emissiveStrength: 23 } },
  });
  f.model.meshes[0].primitives.push({ ...f.model.meshes[0].primitives[0], material: 1 });
  const r = await roundtrip(decode(f));
  assert.deepEqual(r.json.extensionsRequired, [EXT]);
  assert.deepEqual(
    r.reloaded.drawables.map((d) => d.emissiveFactor),
    [
      [1, 2, 4],
      [0, 23, 0],
    ],
  );
});

test("HDR export validates every material before starting encoded image resolution", async () => {
  for (const rgb of [
    [-1, 0, 0],
    [NaN, 0, 0],
    [Infinity, 0, 0],
    [1e100, 0, 0],
  ]) {
    const f = frame(decode(maps(fixture())));
    f.entries.push({
      ...f.entries[0],
      drawable: { ...f.entries[0].drawable, emissiveFactor: rgb },
    });
    let calls = 0;
    await assert.rejects(
      exportAnimationPoseGLB(f.pose, f.entries, {
        resolveTexture: () => {
          calls++;
          return encoded();
        },
      }),
      { code: "ANIMATION_EXPORT_VALUE" },
    );
    assert.equal(calls, 0);
  }
});

test("model export captures current HDR edits before awaiting images without blocking later pose changes", async () => {
  const decoded = decode(maps(fixture())),
    f = frame(decoded);
  const exporter = createAnimationModelExporter(
    f.pose,
    decoded.drawables,
    decoded.source,
    true,
    f.entries.map((e) => e.deformer),
  );
  decoded.drawables[0].emissiveFactor = [16, 8, 4];
  let resolve;
  const pending = exporter.exportPoseGLB({
    resolveTexture: () =>
      new Promise((r) => {
        resolve = r;
      }),
  });
  decoded.drawables[0].emissiveFactor.fill(0);
  f.pose.version++;
  exporter.dispose();
  // Two distinct color-space views: release the first, then the second callback.
  resolve(encoded());
  await Promise.resolve();
  resolve(encoded());
  const r = unpack(await pending),
    m = r.json.materials[0];
  assert.equal(m.extensions[EXT].emissiveStrength, 16);
  assert.deepEqual(m.emissiveFactor, [1, 0.5, 0.25]);
  assert.equal(f.pose.disposed, false);
});
