import assert from "node:assert/strict";
import test from "node:test";
import { exportAnimationPoseGLB } from "./animation_pose_export.mjs";

// Structurally valid KTX2 containers with synthetic payload bytes. These tests
// exercise the real GLB writer/header preflight, not a Basis decoder or GPU.
function ktx({
  colorSpace = "srgb",
  model = 166,
  scheme = model === 163 ? 1 : 0,
  levels = 3,
} = {}) {
  const dfd = 80 + 24 * levels,
    sgd = model === 163 ? Math.ceil((dfd + 44) / 8) * 8 : 0,
    sgdLength = sgd ? 24 : 0;
  let size = Math.ceil((sgd ? sgd + sgdLength : dfd + 44) / 16) * 16,
    w = 4;
  const mips = [];
  for (let i = 0; i < levels; i++) {
    const raw = Math.ceil(w / 4) ** 2 * 16,
      length = scheme === 0 ? raw : 8;
    mips.push({ offset: size, length, raw: model === 163 ? 0 : raw });
    size += length;
    w = Math.max(1, Math.floor(w / 2));
  }
  const b = new Uint8Array(size),
    v = new DataView(b.buffer),
    u = (at, n) => v.setUint32(at, n, true),
    u64 = (at, n) => v.setBigUint64(at, BigInt(n), true);
  b.set([171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10]);
  for (const [at, n] of [
    [16, 1],
    [20, 4],
    [24, 4],
    [36, 1],
    [40, levels],
    [44, scheme],
    [48, dfd],
    [52, 44],
  ])
    u(at, n);
  u64(64, sgd);
  u64(72, sgdLength);
  mips.forEach((m, i) => {
    u64(80 + i * 24, m.offset);
    u64(88 + i * 24, m.length);
    u64(96 + i * 24, m.raw);
    b.fill(0x70 + i, m.offset, m.offset + m.length);
  });
  u(dfd, 44);
  v.setUint16(dfd + 8, 2, true);
  v.setUint16(dfd + 10, 40, true);
  b.set(
    [
      model,
      colorSpace === "srgb" ? 1 : 0,
      colorSpace === "srgb" ? 2 : 1,
      0,
      3,
      3,
      0,
      0,
      model === 166 ? 16 : 0,
    ],
    dfd + 12,
  );
  return b;
}
const png = Uint8Array.from(
  Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==",
    "base64",
  ),
);
const texture = () => ({ view: {}, sampler: {} });
function fixture(drawable = {}) {
  return {
    pose: { version: 7, disposed: false },
    entry: {
      deformer: {
        vertexCount: 3,
        disposed: false,
        poseVersion: 7,
        worldMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1],
        positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
      },
      source: { node: 2, mesh: 0, primitive: 0, material: 1 },
      drawable: { shading: "unlit", texCoords: Float32Array.of(0, 0, 1, 0, 0, 1), ...drawable },
    },
  };
}
async function write(drawable, resolveTexture, options = {}) {
  const { pose, entry } = fixture(drawable);
  return parse(await exportAnimationPoseGLB(pose, [entry], { resolveTexture, ...options }));
}
// Independent GLB layout reader: no production parser is used as its own oracle.
function parse(buffer) {
  const v = new DataView(buffer),
    bytes = new Uint8Array(buffer);
  assert.equal(v.getUint32(0, true), 0x46546c67);
  assert.equal(v.getUint32(4, true), 2);
  assert.equal(v.getUint32(8, true), bytes.length);
  const length = v.getUint32(12, true);
  assert.equal(length % 4, 0);
  assert.equal(v.getUint32(16, true), 0x4e4f534a);
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
  assert.equal(v.getUint32(24 + length, true), 0x004e4942);
  const bin = bytes.subarray(28 + length);
  assert.equal(bin.length, v.getUint32(20 + length, true));
  assert.ok(bin.length - json.buffers[0].byteLength < 4);
  const view = (i) => {
    const b = json.bufferViews[i];
    assert.equal(b.buffer, 0);
    assert.equal(b.byteOffset % 4, 0);
    assert.ok(b.byteOffset + b.byteLength <= json.buffers[0].byteLength);
    return bin.subarray(b.byteOffset, b.byteOffset + b.byteLength);
  };
  const image = (i) => view(json.images[i].bufferView);
  return { json, bin, view, image, buffer };
}
const encoded = (bytes) => ({ bytes, mimeType: "image/ktx2" });
const exportError = { code: "ANIMATION_EXPORT_TEXTURE" };

for (const options of [{}, { scheme: 2 }, { model: 163 }])
  test(`preserves ${JSON.stringify(options)} KTX2 payload and all mip data in required BasisU extension`, async () => {
    const bytes = ktx(options),
      copy = bytes.slice(),
      t = texture();
    let calls = 0;
    const result = await write({ baseColorTexture: t }, (descriptor, context) => {
      calls++;
      assert.equal(descriptor.view, t.view);
      assert.deepEqual(context.colorSpaces, ["srgb"]);
      assert.ok(Object.isFrozen(context.colorSpaces));
      return {
        ...encoded(bytes),
        sampler: { wrapS: 33071, wrapT: 33648, minFilter: 9987, magFilter: 9728 },
      };
    });
    assert.equal(calls, 1);
    assert.deepEqual(result.image(0), copy);
    assert.deepEqual(bytes, copy);
    assert.deepEqual(result.json.textures, [
      { sampler: 0, extensions: { KHR_texture_basisu: { source: 0 } } },
    ]);
    assert.deepEqual(result.json.extensionsUsed, ["KHR_materials_unlit", "KHR_texture_basisu"]);
    assert.deepEqual(result.json.extensionsRequired, result.json.extensionsUsed);
    assert.equal(result.json.images[0].mimeType, "image/ktx2");
    assert.equal(result.json.images[0].uri, undefined);
    assert.equal(result.json.bufferViews[result.json.images[0].bufferView].target, undefined);
    assert.deepEqual(result.json.samplers[0], {
      wrapS: 33071,
      wrapT: 33648,
      magFilter: 9728,
      minFilter: 9987,
    });
    assert.equal(result.json.skins, undefined);
    assert.equal(result.json.animations, undefined);
  });
test("linear normal and metallic-roughness maps preserve separate coordinates and normal scale", async () => {
  const t = texture(),
    bytes = ktx({ colorSpace: "linear" });
  let calls = 0;
  const r = await write(
    {
      shading: "metallic-roughness",
      normalTexture: t,
      metallicRoughnessTexture: t,
      normalScale: 0.25,
      mapCoordinates: { normalTexture: { uvTransform: [2, 0, 0, 3, 0.1, 0.2] } },
    },
    (d, { colorSpaces }) => {
      calls++;
      assert.deepEqual(colorSpaces, ["linear"]);
      return encoded(bytes);
    },
  );
  assert.equal(calls, 1);
  assert.equal(r.json.textures.length, 1);
  const material = r.json.materials[0];
  assert.equal(material.normalTexture.scale, 0.25);
  assert.equal(material.pbrMetallicRoughness.metallicRoughnessTexture.texCoord, 0);
  assert.equal(material.normalTexture.texCoord, 1);
  assert.deepEqual(r.json.extensionsRequired, ["KHR_texture_basisu"]);
});
test("shared base and emissive texture is resolved only once with one color contract", async () => {
  const t = texture();
  let calls = 0;
  const r = await write(
    { shading: "metallic-roughness", baseColorTexture: t, emissiveTexture: t },
    (d, { colorSpaces }) => {
      calls++;
      assert.deepEqual(colorSpaces, ["srgb"]);
      return encoded(ktx());
    },
  );
  assert.equal(calls, 1);
  assert.equal(r.json.images.length, 1);
  assert.equal(r.json.materials[0].emissiveTexture.index, 0);
});
test("mixed PNG, JPEG and KTX2 images retain their own source routes", async () => {
  const { pose, entry } = fixture({ baseColorTexture: texture() }),
    two = fixture({ baseColorTexture: texture() }).entry,
    three = fixture({ baseColorTexture: texture() }).entry;
  const jpeg = Uint8Array.of(255, 216, 255, 217),
    sources = [
      { bytes: png, mimeType: "image/png" },
      encoded(ktx()),
      { bytes: jpeg, mimeType: "image/jpeg" },
    ];
  let at = 0;
  const r = parse(
    await exportAnimationPoseGLB(pose, [entry, two, three], {
      resolveTexture: () => sources[at++],
    }),
  );
  assert.equal(r.json.textures[0].source, 0);
  assert.equal(r.json.textures[1].source, undefined);
  assert.equal(r.json.textures[1].extensions.KHR_texture_basisu.source, 1);
  assert.equal(r.json.textures[2].source, 2);
  assert.deepEqual(r.image(0), png);
  assert.deepEqual(r.image(2), jpeg);
  assert.equal(r.json.extensionsRequired.filter((n) => n === "KHR_texture_basisu").length, 1);
});
test("different samplers share unchanged encoded bytes but remain distinct textures", async () => {
  const a = texture(),
    b = { view: a.view, sampler: {} },
    bytes = ktx();
  let calls = 0;
  const r = await write(
    { shading: "metallic-roughness", baseColorTexture: a, emissiveTexture: b },
    () => ({ ...encoded(bytes), sampler: { wrapS: ++calls === 1 ? 33071 : 10497 } }),
  );
  assert.equal(r.json.images.length, 1);
  assert.equal(r.json.textures.length, 2);
  assert.equal(r.json.samplers.length, 2);
  assert.equal(r.json.textures[0].extensions.KHR_texture_basisu.source, 0);
  assert.equal(r.json.textures[1].extensions.KHR_texture_basisu.source, 0);
});
test("reused mutable encoded arena keeps distinct snapshots instead of corrupting image identity", async () => {
  const bytes = ktx(),
    first = bytes.slice(),
    a = texture(),
    b = texture();
  let calls = 0;
  const r = await write(
    { shading: "metallic-roughness", baseColorTexture: a, emissiveTexture: b },
    () => {
      if (calls++) bytes[bytes.length - 1] = 0x33;
      return encoded(bytes);
    },
  );
  assert.equal(r.json.images.length, 2);
  assert.deepEqual(r.image(0), first);
  assert.deepEqual(r.image(1), bytes);
  assert.notEqual(
    r.json.textures[0].extensions.KHR_texture_basisu.source,
    r.json.textures[1].extensions.KHR_texture_basisu.source,
  );
});
test("reused PNG arenas retain each distinct observed value too", async () => {
  const bytes = png.slice(),
    first = bytes.slice();
  let calls = 0;
  const r = await write(
    { shading: "metallic-roughness", baseColorTexture: texture(), emissiveTexture: texture() },
    () => {
      if (calls++) bytes[bytes.length - 1] ^= 1;
      return { bytes, mimeType: "image/png" };
    },
  );
  assert.equal(r.json.images.length, 2);
  assert.deepEqual(r.image(0), first);
  assert.deepEqual(r.image(1), bytes);
  assert.equal(r.json.extensionsRequired, undefined);
});
test("offset KTX2 views embed only the selected byte range", async () => {
  const bytes = ktx(),
    arena = new Uint8Array(bytes.length + 24);
  arena.fill(0xfe);
  arena.set(bytes, 12);
  const r = await write({ baseColorTexture: texture() }, () =>
    encoded(arena.subarray(12, 12 + bytes.length)),
  );
  assert.deepEqual(r.image(0), bytes);
});
test("KTX2 color use cannot be silently changed or shared across incompatible maps", async () => {
  const t = texture();
  await assert.rejects(
    write({ shading: "metallic-roughness", normalTexture: t }, () => encoded(ktx())),
    exportError,
  );
  await assert.rejects(
    write({ baseColorTexture: t }, () => encoded(ktx({ colorSpace: "linear" }))),
    exportError,
  );
  await assert.rejects(
    write({ shading: "metallic-roughness", baseColorTexture: t, normalTexture: t }, () =>
      encoded(ktx()),
    ),
    exportError,
  );
});
test("linear KTX2 uses unspecified primaries, not merely a linear transfer marker", async () => {
  const bytes = ktx({ colorSpace: "linear" }),
    dfd = new DataView(bytes.buffer).getUint32(48, true);
  bytes[dfd + 13] = 1;
  await assert.rejects(
    write({ shading: "metallic-roughness", normalTexture: texture() }, () => encoded(bytes)),
    exportError,
  );
});
test("invalid signatures, truncation, nondefault orientation and raw pixels fail rather than dropping maps", async () => {
  for (const bytes of [new Uint8Array(64), png, ktx().subarray(0, 100)])
    await assert.rejects(
      write({ baseColorTexture: texture() }, () => encoded(bytes)),
      (e) => e.code?.startsWith("GLTF_KTX2_"),
    );
  const bytes = ktx(),
    dfd = new DataView(bytes.buffer).getUint32(48, true);
  bytes[dfd + 15] = 1;
  await assert.rejects(
    write({ baseColorTexture: texture() }, () => encoded(bytes)),
    { code: "GLTF_KTX2_PROFILE" },
  );
  await assert.rejects(
    write({ baseColorTexture: texture() }, () => ({ bytes: ktx(), mimeType: "image/png" })),
    exportError,
  );
});
test("unsafe encoded storage and bad sampler definitions fail", async () => {
  await assert.rejects(
    write({ baseColorTexture: texture() }, () =>
      encoded(new Uint8Array(new SharedArrayBuffer(100))),
    ),
    { code: "ANIMATION_EXPORT_STORAGE" },
  );
  const bytes = new Uint8Array(new ArrayBuffer(100, { maxByteLength: 200 }));
  await assert.rejects(
    write({ baseColorTexture: texture() }, () => encoded(bytes)),
    { code: "ANIMATION_EXPORT_STORAGE" },
  );
  await assert.rejects(
    write({ baseColorTexture: texture() }, () => ({
      ...encoded(ktx()),
      sampler: { minFilter: 0 },
    })),
    exportError,
  );
});
test("exact final-file byte budget is enforced including compressed image and JSON metadata", async () => {
  const bytes = ktx(),
    drawable = { baseColorTexture: texture() },
    r = await write(drawable, () => encoded(bytes));
  const exact = await write(drawable, () => encoded(bytes), { maxBytes: r.buffer.byteLength });
  assert.equal(exact.buffer.byteLength, r.buffer.byteLength);
  await assert.rejects(
    write(drawable, () => encoded(bytes), { maxBytes: r.buffer.byteLength - 1 }),
    { code: "ANIMATION_EXPORT_LIMIT" },
  );
  await assert.rejects(
    write(drawable, () => encoded(bytes), { maxBytes: 128 }),
    { code: "ANIMATION_EXPORT_LIMIT" },
  );
});
test("all geometry is captured before image awaits and survives later model disposal", async () => {
  const { pose, entry } = fixture({ baseColorTexture: texture() });
  let done;
  const pending = exportAnimationPoseGLB(pose, [entry], {
    resolveTexture: () =>
      new Promise((resolve) => {
        done = resolve;
      }),
  });
  entry.deformer.positions.fill(100);
  pose.version++;
  pose.disposed = true;
  entry.deformer.disposed = true;
  done(encoded(ktx()));
  const r = parse(await pending),
    position = r.json.accessors[r.json.meshes[0].primitives[0].attributes.POSITION];
  assert.deepEqual(position.min, [3, 4, 5]);
  assert.deepEqual(position.max, [4, 5, 5]);
  assert.equal(r.json.extras.f3d.poseVersion, 7);
});
test("pre-abort avoids resolution and mid-await abort rejects without owning the texture provider", async () => {
  const c = new AbortController();
  c.abort();
  let calls = 0;
  await assert.rejects(
    write(
      { baseColorTexture: texture() },
      () => {
        calls++;
      },
      { signal: c.signal },
    ),
    { name: "AbortError" },
  );
  assert.equal(calls, 0);
  const next = new AbortController();
  let reject;
  const pending = write(
    { baseColorTexture: texture() },
    () =>
      new Promise((r, bad) => {
        reject = bad;
      }),
    { signal: next.signal },
  );
  next.abort();
  await assert.rejects(pending, { name: "AbortError" });
  reject(new Error("late provider error"));
});
test("untextured output retains reflected winding and no texture extensions", async () => {
  const { pose, entry } = fixture();
  entry.deformer.worldMatrix[0] = -1;
  const r = parse(await exportAnimationPoseGLB(pose, [entry]));
  assert.equal(r.json.textures, undefined);
  assert.deepEqual(r.json.extensionsRequired, ["KHR_materials_unlit"]);
  const a = r.json.accessors[r.json.meshes[0].primitives[0].indices];
  assert.deepEqual([...new Uint16Array(r.view(a.bufferView).slice().buffer)], [0, 2, 1]);
});
test("PNG remains legal across color/data uses and adds no BasisU declaration", async () => {
  const t = texture(),
    r = await write(
      { shading: "metallic-roughness", baseColorTexture: t, normalTexture: t },
      (d, { colorSpaces }) => {
        assert.deepEqual(colorSpaces, ["srgb", "linear"]);
        return { bytes: png, mimeType: "image/png" };
      },
    );
  assert.equal(r.json.images.length, 1);
  assert.equal(r.json.extensionsRequired, undefined);
  assert.equal(r.json.textures[0].source, 0);
});
