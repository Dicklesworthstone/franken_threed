import assert from "node:assert/strict";
import test from "node:test";
import { createCpuGltfAnimationModel, decodeGltfAnimationModel, prepareGltfAnimationModel } from "./animation_model.mjs";
import { createGpuGltfAnimationScene, createGpuDecodedAnimationScene } from "./animation_model_gpu.mjs";
import { loadGpuGltfAnimationScene } from "./gltf_scene_loader.mjs";
import { loadGltfAsset } from "./gltf_asset.mjs";
import { materialVariantFixture as fixture, VARIANTS as EXT } from "./fixtures/animation/material_variants_fixture.mjs";
import { textureDevice } from "./fixtures/gpu_texture_device.mjs";
import { imageBytes, pngBase64 } from "./fixtures/animation/image_fixture.mjs";
import { gltfImageDimensions } from "./gltf_textures.mjs";

const code = expected => error => error.code === expected;
const resource = () => ({ view: {}, sampler: {} });
const utf8 = json => new TextEncoder().encode(JSON.stringify(json));
const close = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-6, `${value} != ${expected[i]}`));
};
const selectedUniforms = device => {
  const draw = device.snapshots.findLast(draws => draws.length)?.[0];
  assert.ok(draw, "A real production draw was submitted");
  const binding = draw.groups.get(0), buffer = binding.group.entries[0].resource.buffer;
  const offset = binding.offsets[0] ?? 0;
  return new Float32Array(draw.contents.get(buffer).slice(offset, offset + 256).buffer);
};
const metadata = value => {
  assert.deepEqual(value.materialVariants, [{ index: 0, name: "blue" }, { index: 1, name: "plain" }]);
  assert.equal(value.materialVariant, 0);
  assert.ok(Object.isFrozen(value.materialVariants) && Object.isFrozen(value.materialVariants[0]));
};
// Only WebGPU native calls are recorded. Every production model, texture,
// controller, scene, compute and renderer module runs unchanged. No WGSL executes.
function gpu() {
  const d = textureDevice(), encode = d.createCommandEncoder;
  Object.assign(d.limits, { maxBindingsPerBindGroup: 1000, maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupsPerDimension: 65535 });
  d.createComputePipelineAsync = async x => ({ ...x, getBindGroupLayout: () => ({}) });
  d.createCommandEncoder = () => ({ ...encode(), beginComputePass: () => ({
    setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
  }) });
  return d;
}
const frame = () => ({ colorView: {}, depthView: {}, lighting: { cameraPosition: [0, 0, 3], lights: [] },
  viewProjection: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
function loading(f) {
  const requests = [], bitmaps = [], png = imageBytes(pngBase64);
  return { requests, bitmaps, options: {
    assets: { baseURL: "https://variants.invalid/model.gltf", fetch: async url => {
      requests.push(url);
      assert.ok(["https://variants.invalid/red.png", "https://variants.invalid/blue.png"].includes(url));
      return new Response(png);
    } },
    textures: { createImageBitmap: async blob => {
      const bitmap = { ...gltfImageDimensions(new Uint8Array(await blob.arrayBuffer()), blob.type),
        closed: false, close() { this.closed = true; } };
      bitmaps.push(bitmap); return bitmap;
    } },
    decode: { materialVariant: "blue" }, scene: { maxBytes: 16384 },
  } };
}

for (const selection of [0, "blue"])
  test(`real decoder resolves ${JSON.stringify(selection)} to the complete selected material`, () => {
    const f = fixture(), before = structuredClone(f.json);
    const decoded = decodeGltfAnimationModel(f.json, f.buffers, { materialVariant: selection });
    metadata(decoded);
    assert.deepEqual(decoded.drawables[0].baseColor, [0, 0, 1, 0.5]);
    assert.equal(decoded.drawables[0].clearcoatFactor, 0.75);
    assert.equal(decoded.drawables[0].metallicFactor, 0.25);
    assert.equal(decoded.drawables[0].alphaMode, "BLEND");
    assert.equal(decoded.source[0].material, 1);
    assert.deepEqual(f.json, before);
  });
for (const selection of [null, 1, "plain"])
  test(`real default material survives selection ${JSON.stringify(selection)}`, () => {
    const f = fixture();
    const decoded = decodeGltfAnimationModel(f.json, f.buffers, { materialVariant: selection });
    assert.deepEqual(decoded.drawables[0].baseColor, [1, 0, 0, 1]);
    assert.equal(decoded.source[0].material, 0);
  });
test("implicit material is not silently replaced with material zero", () => {
  const f = fixture(); delete f.primitive.material;
  const decoded = decodeGltfAnimationModel(f.json, f.buffers);
  assert.deepEqual(decoded.drawables[0].baseColor, [1, 1, 1, 1]);
  assert.equal(decoded.source[0].material, null);
});
test("selected texture and independent UV/color-space requirements reach the resolver", () => {
  const f = fixture({ textured: true }), requests = [];
  f.json.materials[1].normalTexture = { index: 1 };
  const plan = prepareGltfAnimationModel(f.json, f.buffers, { materialVariant: "blue" });
  metadata(plan);
  assert.deepEqual(plan.textureRequests.map(r => [r.imageIndex, r.colorSpace]), [[1, "srgb"], [1, "linear"]]);
  const decoded = plan.resolveTextures(r => { requests.push(r); return resource(); });
  assert.equal(requests.length, 2);
  const drawable = decoded.drawables[0];
  close(drawable.mapCoordinates.baseColorTexture.texCoords, [0.25, 0.5, 0.75, 0.5, 0.25, 1]);
  close(drawable.mapCoordinates.baseColorTexture.uvTransform, [1, 0, 0, 1, 0.5, 0.25]);
  close(drawable.mapCoordinates.normalTexture.texCoords, [0, 0, 1, 0, 0, 1]);
  assert.notEqual(drawable.baseColorTexture, drawable.normalTexture);
});
test("prepared selection, materials, names and geometry survive caller edits and resolver retry", () => {
  const f = fixture({ textured: true });
  const plan = prepareGltfAnimationModel(f.json, f.buffers, { materialVariant: "blue" });
  f.json.extensions[EXT].variants[0].name = "changed";
  f.json.materials[1].pbrMetallicRoughness.baseColorFactor.fill(99);
  f.primitive.extensions[EXT].mappings[0].material = 0;
  f.buffers.forEach(b => b.fill(99));
  assert.throws(() => plan.resolveTextures(() => { throw Error("not ready"); }), /not ready/);
  const decoded = plan.resolveTextures(resource);
  metadata(decoded);
  close(decoded.drawables[0].geometry.positions, [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepEqual(decoded.drawables[0].baseColor, [0, 0, 1, 0.5]);
  assert.equal(decoded.source[0].material, 1);
  assert.throws(() => plan.resolveTextures(resource), code("GLTF_MODEL_PREPARED"));
});
for (const [name, mutate, options, error] of [
  ["duplicate mapping", f => { f.primitive.extensions[EXT].mappings.push({ material: 0, variants: [0] }); }, {}, "GLTF_VARIANT_DUPLICATE"],
  ["invalid unselected mapping", f => { f.primitive.extensions[EXT].mappings.push({ material: 99, variants: [1] }); }, {}, "GLTF_VARIANT_INDEX"],
  ["unknown selection", () => {}, { materialVariant: "absent" }, "GLTF_VARIANT_SELECTION"],
  ["variant budget", () => {}, { materialVariantLimits: { maxVariants: 1 } }, "GLTF_VARIANT_LIMIT"],
  ["required source-route extension", f => f.json.extensionsRequired.push("OTHER"), { materialVariant: 0 }, "GLTF_MODEL_UNSUPPORTED"],
]) test(`${name} fails before buffer callbacks or texture resolution`, () => {
  const f = fixture({ textured: true }); mutate(f);
  assert.throws(() => decodeGltfAnimationModel(f.json, () => assert.fail("buffer read"), {
    resolveTexture: () => assert.fail("texture resolver"), ...options,
  }), code(error));
});
test("unselected optional material semantics are not decoded, but selecting them rejects", () => {
  const f = fixture();
  f.json.materials[1].extensions.KHR_materials_transmission = { transmissionFactor: 1 };
  assert.equal(decodeGltfAnimationModel(f.json, f.buffers).source[0].material, 0);
  assert.throws(() => decodeGltfAnimationModel(f.json, f.buffers, { materialVariant: 0 }), code("GLTF_MODEL_UNSUPPORTED"));
});
test("plain assets retain their existing selected-scene behavior without variant metadata", () => {
  const f = fixture(); delete f.json.extensions; delete f.json.extensionsRequired; delete f.json.extensionsUsed;
  delete f.primitive.extensions;
  f.json.meshes.push({ primitives: null }); // Unselected mesh is outside this decode route.
  const decoded = decodeGltfAnimationModel(f.json, f.buffers);
  assert.equal(Object.hasOwn(decoded, "materialVariants"), false);
  assert.deepEqual(decoded.drawables[0].baseColor, [1, 0, 0, 1]);
});
test("independent CPU instances keep different materials and unchanged pose/deformation behavior", () => {
  const f = fixture(), a = createCpuGltfAnimationModel(f.json, f.buffers, { materialVariant: 0 }),
    b = createCpuGltfAnimationModel(f.json, f.buffers);
  try {
    metadata(a);
    a.sample(0.5);
    close(a.deformers[0].worldMatrix.slice(12, 15), [2, 1, 0]);
    close(b.deformers[0].worldMatrix.slice(12, 15), [0, 0, 0]);
    assert.deepEqual(a.drawables[0].baseColor, [0, 0, 1, 0.5]);
    assert.deepEqual(b.drawables[0].baseColor, [1, 0, 0, 1]);
    assert.notEqual(a.pose, b.pose);
  } finally { a.dispose(); b.dispose(); }
});
test("instance expansion retains selected materials and original node identity", () => {
  const f = fixture(), EXT_INSTANCE = "EXT_mesh_gpu_instancing";
  f.json.extensionsRequired.push(EXT_INSTANCE);
  f.json.nodes[0].extensions = { [EXT_INSTANCE]: { attributes: { TRANSLATION: f.attr([0, 0, 0, 3, 0, 0]) } } };
  const decoded = decodeGltfAnimationModel(f.json, f.buffers, { materialVariant: 0 });
  metadata(decoded);
  assert.equal(decoded.drawables.length, 2);
  assert.deepEqual(decoded.source.map(s => s.material), [1, 1]);
  assert.deepEqual(Object.values(decoded.instanceOrigins), [{ node: 0, instance: 0 }, { node: 0, instance: 1 }]);
});
test("CPU current-pose export contains the selected appearance and actual animated positions", async () => {
  const f = fixture(), model = createCpuGltfAnimationModel(f.json, f.buffers, { materialVariant: 0, exporting: true });
  try {
    model.sample(0.5);
    const asset = await loadGltfAsset(await model.exportPoseGLB());
    const decoded = decodeGltfAnimationModel(asset.json, asset.buffers);
    assert.deepEqual(decoded.drawables[0].baseColor, [0, 0, 1, 0.5]);
    assert.equal(decoded.drawables[0].clearcoatFactor, 0.75);
    close(decoded.drawables[0].geometry.positions, [2, 1, 0, 3, 1, 0, 2, 2, 0]);
    assert.equal(asset.json.extensions?.[EXT], undefined);
    assert.equal(asset.json.animations, undefined);
  } finally { model.dispose(); }
});
for (const prepared of [false, true])
  test(`real GPU scene receives selected material and metadata, prepared=${prepared}`, async () => {
    const f = fixture(), d = gpu();
    const model = prepared
      ? await createGpuDecodedAnimationScene(d, decodeGltfAnimationModel(f.json, f.buffers, { materialVariant: 0 }))
      : await createGpuGltfAnimationScene(d, f.json, f.buffers, { decode: { materialVariant: "blue" } });
    try {
      metadata(model);
      model.controller.createAction(0).play(); model.update(0.5); model.render(frame());
      assert.equal(model.source[0].material, 1);
      close(model.pose.worldMatrices.slice(12, 15), [2, 1, 0]);
      close(selectedUniforms(d).slice(16, 20), [0, 0, 1, 0.5]);
      await model.whenIdle();
    } finally { model.dispose(); }
    assert.ok(d.buffers.every(b => b.destroyed));
  });
test("owning loader fetches and uploads only the selected image with the real scene pipeline", async () => {
  const f = fixture({ textured: true }), d = gpu(), io = loading(f);
  const model = await loadGpuGltfAnimationScene(d, utf8(f.json), io.options);
  try {
    metadata(model);
    assert.deepEqual(io.requests, ["https://variants.invalid/blue.png"]);
    assert.equal(d.textures.length, 1);
    model.render(frame()); await model.whenIdle();
    assert.equal(model.source[0].material, 1);
    close(selectedUniforms(d).slice(16, 20), [0, 0, 1, 0.5]);
  } finally { model.dispose(); }
  assert.ok(d.buffers.every(b => b.destroyed) && d.textures.every(t => t.destroyed));
  assert.ok(io.bitmaps.every(b => b.closed));
});
test("construction snapshots nested variant limits before asset fetch yields", async () => {
  const f = fixture(), d = gpu(), limits = { maxVariants: 2 };
  let resolve;
  const response = new Promise(r => { resolve = r; });
  const pending = loadGpuGltfAnimationScene(d, "https://variants.invalid/model.gltf", {
    assets: { fetch: () => response }, decode: { materialVariant: 0, materialVariantLimits: limits },
  });
  limits.maxVariants = 1; resolve(new Response(utf8(f.json)));
  const model = await pending;
  metadata(model); model.dispose();
});
test("invalid selection allocates no textures, GPU buffers or shader pipelines", async () => {
  const f = fixture({ textured: true }), d = gpu(), io = loading(f);
  io.options.decode.materialVariant = "missing";
  await assert.rejects(loadGpuGltfAnimationScene(d, utf8(f.json), io.options), code("GLTF_VARIANT_SELECTION"));
  assert.deepEqual(io.requests, []);
  assert.equal(d.buffers.length + d.textures.length + d.pipelines.length, 0);
});
