import assert from "node:assert/strict";
import test from "node:test";
import { exportAnimationPoseGLB } from "./animation_pose_export.mjs";
import { gltfWebpDimensions } from "./gltf_webp.mjs";
import { webpFixtures, webpBytes } from "./fixtures/animation/webp_fixture.mjs";

const WEBP = "EXT_texture_webp";
const encoded = (fixture = webpFixtures[0]) => ({ bytes: webpBytes(fixture), mimeType: "image/webp" });
const texture = () => ({ view: {}, sampler: {} });
// Fixtures are the exporter's documented CPU deformation outputs. The actual
// exporter writes the GLB; no decoder, GPU, alternate exporter or mocked module.
function fixture() {
  const matrix = Float64Array.of(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1);
  const pose = { version: 7, disposed: false, nodeCount: 1, worldMatrices: matrix };
  const entry = {
    deformer: {
      disposed: false, poseVersion: 7, vertexCount: 3, worldMatrix: matrix,
      positions: Float64Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
      normals: Float64Array.of(0, 0, 1, 0, 0, 1, 0, 0, 1),
    },
    drawable: {
      shading: "metallic-roughness", baseColorTexture: texture(),
      texCoords: Float64Array.of(0, 0, 1, 0, 0, 1),
      alphaMode: "MASK", alphaCutoff: 0.25, doubleSided: true,
    },
    source: { node: 3, mesh: 2, primitive: 1, material: 0 },
  };
  return { pose, entry };
}
const pack = (f, options = {}) => exportAnimationPoseGLB(f.pose, [f.entry], {
  resolveTexture: () => encoded(), ...options,
});
// Independently inspect the emitted GLB structure and byte ranges, rather than
// trusting an exporter-provided success flag or requiring a browser to decode it.
function unpack(buffer) {
  const data = new Uint8Array(buffer), header = new DataView(buffer);
  assert.equal(header.getUint32(0, true), 0x46546c67);
  assert.equal(header.getUint32(4, true), 2);
  assert.equal(header.getUint32(8, true), data.length);
  assert.equal(header.getUint32(16, true), 0x4e4f534a);
  const jsonLength = header.getUint32(12, true), binLength = header.getUint32(20 + jsonLength, true);
  assert.equal(jsonLength % 4, 0); assert.equal(binLength % 4, 0);
  assert.equal(header.getUint32(24 + jsonLength, true), 0x004e4942);
  assert.equal(28 + jsonLength + binLength, data.length);
  const json = JSON.parse(new TextDecoder().decode(data.subarray(20, 20 + jsonLength)));
  const bin = data.subarray(28 + jsonLength);
  assert.ok(bin.length >= json.buffers[0].byteLength);
  assert.ok(bin.length - json.buffers[0].byteLength < 4);
  for (const v of bin.subarray(json.buffers[0].byteLength)) assert.equal(v, 0);
  const viewBytes = index => {
    const view = json.bufferViews[index]; assert.equal(view.buffer, 0);
    const offset = view.byteOffset ?? 0;
    assert.equal(offset % 4, 0); assert.ok(view.byteLength > 0);
    assert.ok(offset + view.byteLength <= json.buffers[0].byteLength);
    return bin.subarray(offset, offset + view.byteLength);
  };
  return { json, viewBytes, image: index => viewBytes(json.images[index].bufferView),
    floats(index) {
      const accessor = json.accessors[index]; assert.equal(accessor.componentType, 5126);
      const bytes = viewBytes(accessor.bufferView);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      return Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
    },
  };
}
for (const image of webpFixtures) test(`posed ${image.name} WebP retains exact bytes and a required extension source`, async () => {
  const f = fixture(), source = encoded(image), before = source.bytes.slice();
  let calls = 0;
  const result = unpack(await pack(f, { resolveTexture(descriptor, options) {
    calls++; assert.equal(descriptor.view, f.entry.drawable.baseColorTexture.view);
    assert.deepEqual(options.colorSpaces, ["srgb"]); assert.ok(Object.isFrozen(options.colorSpaces));
    return source;
  } }));
  assert.equal(calls, 1); assert.deepEqual(source.bytes, before);
  assert.deepEqual(result.json.extensionsUsed, [WEBP]);
  assert.deepEqual(result.json.extensionsRequired, [WEBP]);
  assert.deepEqual(result.json.textures, [{ sampler: 0, extensions: { [WEBP]: { source: 0 } } }]);
  assert.equal(result.json.images[0].mimeType, "image/webp");
  assert.equal(result.json.images[0].uri, undefined);
  assert.deepEqual(result.image(0), before);
  assert.deepEqual(gltfWebpDimensions(result.image(0)), { width: 2, height: 2 });
  const primitive = result.json.meshes[0].primitives[0];
  assert.deepEqual(result.floats(primitive.attributes.POSITION), [5, 6, 7, 6, 6, 7, 5, 7, 7]);
  assert.deepEqual(result.floats(primitive.attributes.TEXCOORD_0), [0, 0, 1, 0, 0, 1]);
  assert.equal(result.json.materials[0].alphaMode, "MASK");
  assert.equal(result.json.materials[0].alphaCutoff, 0.25);
  assert.equal(result.json.animations, undefined); assert.equal(result.json.skins, undefined);
  source.bytes.fill(0); assert.deepEqual(result.image(0), before);
});

test("all eight material maps retain WebP identity, independent UVs and normal scales", async () => {
  const f = fixture(), d = f.entry.drawable, descriptor = d.baseColorTexture;
  for (const field of ["metallicRoughnessTexture", "normalTexture", "emissiveTexture", "occlusionTexture",
    "clearcoatTexture", "clearcoatRoughnessTexture", "clearcoatNormalTexture"]) d[field] = descriptor;
  d.clearcoatFactor = 0.7; d.clearcoatRoughnessFactor = 0.2;
  d.normalScale = 0.4; d.clearcoatNormalScale = 0.8; d.occlusionStrength = 0.3;
  d.mapCoordinates = { clearcoatNormalTexture: { uvTransform: [2, 0, 0, 3, 0.25, 0.5] } };
  let calls = 0;
  const out = unpack(await pack(f, { resolveTexture(_, { colorSpaces }) {
    calls++; assert.deepEqual(colorSpaces, ["srgb", "linear"]); return encoded();
  } }));
  assert.equal(calls, 1); assert.equal(out.json.images.length, 1); assert.equal(out.json.textures.length, 1);
  const material = out.json.materials[0], coat = material.extensions.KHR_materials_clearcoat;
  assert.equal(material.normalTexture.scale, 0.4); assert.equal(coat.clearcoatNormalTexture.scale, 0.8);
  assert.equal(material.occlusionTexture.strength, 0.3);
  const infos = [material.pbrMetallicRoughness.baseColorTexture, material.pbrMetallicRoughness.metallicRoughnessTexture,
    material.normalTexture, material.emissiveTexture, material.occlusionTexture,
    coat.clearcoatTexture, coat.clearcoatRoughnessTexture, coat.clearcoatNormalTexture];
  assert.deepEqual(infos.map(info => info.index), Array(8).fill(0));
  assert.deepEqual(infos.map(info => info.texCoord), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(out.floats(out.json.meshes[0].primitives[0].attributes.TEXCOORD_7), [0.25, 0.5, 2.25, 0.5, 0.25, 3.5]);
  assert.equal(out.json.extensionsRequired.filter(name => name === WEBP).length, 1);
});

test("distinct samplers can share encoded WebP without losing wrap or filtering", async () => {
  const f = fixture(), source = encoded(); f.entry.drawable.emissiveTexture = texture();
  const settings = [
    { wrapS: 33071, wrapT: 33648, minFilter: 9728, magFilter: 9729 },
    { wrapS: 10497, wrapT: 10497, minFilter: 9987, magFilter: 9728 },
  ];
  let calls = 0;
  const out = unpack(await pack(f, { resolveTexture: () => ({ ...source, sampler: settings[calls++] }) }));
  assert.equal(calls, 2); assert.equal(out.json.images.length, 1);
  assert.deepEqual(out.json.samplers, settings);
  assert.deepEqual(out.json.textures.map(t => t.extensions[WEBP].source), [0, 0]);
  assert.deepEqual(out.json.textures.map(t => t.sampler), [0, 1]);
});

test("a reused encoded arena preserves each observed WebP snapshot", async () => {
  const f = fixture(), source = encoded(webpFixtures[3]), first = source.bytes.slice();
  f.entry.drawable.emissiveTexture = texture();
  let calls = 0, second;
  const out = unpack(await pack(f, { resolveTexture() {
    if (calls++) source.bytes[source.bytes.length - 3] = 121; // Change only XMP text, x -> y.
    second = source.bytes.slice(); return source;
  } }));
  source.bytes.fill(0);
  assert.equal(out.json.images.length, 2); assert.notDeepEqual(first, second);
  assert.deepEqual(out.image(0), first); assert.deepEqual(out.image(1), second);
});

test("WebP and core PNG keep different glTF texture representations", async () => {
  const f = fixture(); f.entry.drawable.emissiveTexture = texture();
  const png = Uint8Array.from(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGUlEQVR4nAXBAQ0AAAzDIJbcv+UeRNJNwgM+/wYAhdCTnAAAAABJRU5ErkJggg==", "base64"));
  let calls = 0;
  const out = unpack(await pack(f, { resolveTexture: () => calls++ ? { bytes: png, mimeType: "image/png" } : encoded() }));
  assert.equal(out.json.textures[0].source, undefined);
  assert.deepEqual(out.json.textures[0].extensions, { [WEBP]: { source: 0 } });
  assert.equal(out.json.textures[1].source, 1); assert.equal(out.json.textures[1].extensions, undefined);
  assert.deepEqual(out.image(1), png);
});

for (const [name, change] of [
  ["bad signature", bytes => { bytes[0] = 0; }],
  ["wrong extent", bytes => { bytes[4] ^= 2; }],
  ["invalid chunk size", bytes => { new DataView(bytes.buffer).setUint32(16, 0xffffffff, true); }],
]) test(`posed export rejects WebP ${name}`, async () => {
  const source = encoded(); change(source.bytes);
  await assert.rejects(pack(fixture(), { resolveTexture: () => source }), { code: "GLTF_TEXTURE_IMAGE" });
});
for (const [name, flag] of [["animation", 2], ["ICC", 32], ["EXIF", 8]])
  test(`posed export refuses ${name}-dependent WebP rather than changing material meaning`, async () => {
    const source = encoded(webpFixtures[3]); source.bytes[20] |= flag;
    await assert.rejects(pack(fixture(), { resolveTexture: () => source }), { code: "GLTF_TEXTURE_IMAGE" });
  });
test("every truncated WebP prefix is rejected, not embedded as a valid posed texture", async () => {
  const source = encoded();
  for (let end = 0; end < source.bytes.length; end++)
    await assert.rejects(pack(fixture(), { resolveTexture: () => ({ ...source, bytes: source.bytes.subarray(0, end) }) }),
      { code: "GLTF_TEXTURE_IMAGE" });
});
test("WebP MIME cannot disguise PNG and WebP cannot masquerade as core PNG", async () => {
  await assert.rejects(pack(fixture(), { resolveTexture: () => ({ bytes: Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), mimeType: "image/webp" }) }),
    { code: "GLTF_TEXTURE_IMAGE" });
  await assert.rejects(pack(fixture(), { resolveTexture: () => ({ ...encoded(), mimeType: "image/png" }) }),
    { code: "ANIMATION_EXPORT_TEXTURE" });
});

test("WebP export rejects shared, resizable and detached storage", async () => {
  const source = encoded().bytes, detached = source.slice();
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  const shared = new Uint8Array(new SharedArrayBuffer(source.length)); shared.set(source);
  const resizable = new Uint8Array(new ArrayBuffer(source.length, { maxByteLength: source.length + 2 })); resizable.set(source);
  for (const bytes of [shared, resizable, detached])
    await assert.rejects(pack(fixture(), { resolveTexture: () => ({ bytes, mimeType: "image/webp" }) }),
      { code: "ANIMATION_EXPORT_STORAGE" });
});
test("WebP byte-offset views retain only their encoded image range", async () => {
  const source = encoded(), padded = new Uint8Array(source.bytes.length + 17).fill(255);
  padded.set(source.bytes, 7);
  const out = unpack(await pack(fixture(), { resolveTexture: () => ({ ...source, bytes: padded.subarray(7, 7 + source.bytes.length) }) }));
  assert.deepEqual(out.image(0), source.bytes);
});
test("WebP export enforces complete-file and encoded-byte budgets", async () => {
  const f = fixture(), full = await pack(f);
  assert.equal((await pack(f, { maxBytes: full.byteLength })).byteLength, full.byteLength);
  await assert.rejects(pack(f, { maxBytes: full.byteLength - 1 }), { code: "ANIMATION_EXPORT_LIMIT" });
  // Oversized bytes must fail budget admission before even inspecting a header.
  await assert.rejects(pack(f, { maxBytes: 1024, resolveTexture: () => ({ bytes: new Uint8Array(2048), mimeType: "image/webp" }) }),
    { code: "ANIMATION_EXPORT_LIMIT" });
});

test("pose/material snapshot remains valid when live state changes during WebP resolution", async () => {
  const f = fixture(); let finish, start;
  const started = new Promise(resolve => { start = resolve; });
  const pending = pack(f, { resolveTexture() { start(); return new Promise(resolve => { finish = resolve; }); } });
  await started;
  f.entry.deformer.positions.fill(999); f.entry.drawable.texCoords.fill(999);
  f.entry.drawable.alphaCutoff = 0.9; f.pose.version++; f.pose.disposed = true; f.entry.deformer.disposed = true;
  finish(encoded());
  const out = unpack(await pending), attributes = out.json.meshes[0].primitives[0].attributes;
  assert.deepEqual(out.floats(attributes.POSITION), [5, 6, 7, 6, 6, 7, 5, 7, 7]);
  assert.deepEqual(out.floats(attributes.TEXCOORD_0), [0, 0, 1, 0, 0, 1]);
  assert.equal(out.json.materials[0].alphaCutoff, 0.25); assert.equal(out.json.extras.f3d.poseVersion, 7);
});
for (const lateReject of [false, true]) test(`WebP export cancellation observes late ${lateReject ? "rejection" : "resolution"}`, { timeout: 2000 }, async () => {
  const controller = new AbortController(), reason = new Error("cancel WebP pose");
  let finish, reject, start;
  const started = new Promise(resolve => { start = resolve; });
  const pending = pack(fixture(), { signal: controller.signal,
    resolveTexture() { start(); return new Promise((yes, no) => { finish = yes; reject = no; }); },
  });
  await started; controller.abort(reason);
  await assert.rejects(pending, error => error === reason);
  if (lateReject) reject(new Error("late provider failure")); else finish(encoded());
  await new Promise(resolve => setImmediate(resolve));
});
test("already-aborted WebP export does not invoke image providers", async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(pack(fixture(), { signal: controller.signal, resolveTexture: () => assert.fail("No image I/O") }), { name: "AbortError" });
});

test("WebP export does not dispose borrowed texture resources or invoke codecs", async () => {
  const f = fixture();
  f.entry.drawable.baseColorTexture.view.destroy = () => assert.fail("Borrowed GPU texture");
  f.entry.drawable.baseColorTexture.sampler.dispose = () => assert.fail("Borrowed sampler");
  const out = unpack(await pack(f)); assert.equal(out.json.images[0].mimeType, "image/webp");
});
