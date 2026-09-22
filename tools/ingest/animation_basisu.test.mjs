import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Execute the unchanged production material/preparation module. Geometry, pose,
// scene-view and unused query/export imports are explicit boundary doubles: this
// suite tests source selection/material semantics, not accessor or codec decoding.
// The normal test command needs no VM flag; direct flagged runs show each case.
if (!process.execArgv.includes("--experimental-vm-modules")) {
  test("BasisU model source-selection regressions", () => {
    // Do not inherit the parent runner's private IPC/reporting context.
    const env = { ...process.env };
    delete env.NODE_TEST_CONTEXT;
    const out = execFileSync(
      process.execPath,
      [
        "--experimental-vm-modules",
        "--test",
        "--test-reporter=tap",
        fileURLToPath(import.meta.url),
      ],
      { env, encoding: "utf8", timeout: 15000, stdio: ["ignore", "pipe", "pipe"] },
    );
    assert.match(out, /# fail 0/);
    assert.match(out, /# pass [1-9]\d*/);
  });
} else {
  const vm = await import("node:vm");
  class PoseError extends Error {
    constructor(code, message) {
      super(message);
      this.code = code;
    }
  }
  let decodes = 0;
  const unused = () => assert.fail("Unexpected call outside material preparation");
  const uv = Float64Array.of(0, 0, 1, 0, 0, 1);
  const dependencies = {
    "./animation_gltf.mjs": {
      decodeGltfAnimation: () => {
        decodes++;
        return { nodes: [] };
      },
    },
    "./animation_geometry.mjs": {
      decodeGltfGeometry: (model) => ({
        scene: 0,
        diagnostics: [],
        primitives: model.materials.map((_, material) => ({
          node: material,
          mesh: 0,
          primitive: material,
          material,
          indices: Uint32Array.of(0, 1, 2),
          geometry: { positions: Float64Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0) },
          attributes: {
            TEXCOORD_0: { values: uv },
            TEXCOORD_1: { values: Float64Array.from(uv, (x) => x + 2) },
          },
        })),
      }),
    },
    "./animation_runtime.mjs": { AnimationPoseError: PoseError, createAnimationPlayer: unused },
    "./animation_deformer.mjs": { createAnimationDeformer: unused },
    "./animation_model_export.mjs": {
      createAnimationModelExporter: unused,
      AnimationExportError: PoseError,
    },
    "./animation_model_pick.mjs": {
      createAnimationModelPicker: unused,
      AnimationRaycastError: PoseError,
    },
    "./gltf_scene_view.mjs": {
      decodeGltfSceneView: () => ({ cameras: [], lights: [] }),
      createGltfSceneView: unused,
      GltfSceneViewError: PoseError,
    },
  };
  const source = await readFile(new URL("./animation_model.mjs", import.meta.url), "utf8");
  const module = new vm.SourceTextModule(source);
  await module.link((specifier) => {
    const exports = dependencies[specifier];
    assert.ok(exports, "Unexpected import: " + specifier);
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    });
  });
  await module.evaluate();
  const { prepareGltfAnimationModel: prepare, decodeGltfAnimationModel: decode } = module.namespace;
  const ext = "KHR_texture_basisu";
  const fixture = () => ({
    asset: { version: "2.0" },
    extensionsUsed: [ext],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0, extensions: { [ext]: { source: 1 } }, sampler: 0 }],
    images: [{ uri: "fallback.png", mimeType: "image/png" }, { uri: "compressed.ktx2" }],
    samplers: [{ wrapS: 33071, wrapT: 33648, minFilter: 9984, magFilter: 9728 }],
  });
  const resource = () => ({ view: {}, sampler: {} });

  test("optional extension defaults to authored fallback without changing source JSON", () => {
    const json = fixture(),
      before = structuredClone(json),
      plan = prepare(json, []);
    assert.equal(plan.textureRequests[0].imageIndex, 0);
    assert.equal(plan.textureRequests[0].image.uri, "fallback.png");
    assert.deepEqual(json, before);
    assert.equal(prepare(json, [], { basisu: false }).textureRequests[0].imageIndex, 0);
  });
  test("enabled BasisU selects extension source and preserves texture/sampler identity", () => {
    const json = fixture(),
      before = structuredClone(json),
      plan = prepare(json, [], { basisu: true });
    const r = plan.textureRequests[0];
    assert.equal(r.imageIndex, 1);
    assert.equal(r.textureIndex, 0);
    assert.equal(r.colorSpace, "srgb");
    assert.deepEqual(r.image, { uri: "compressed.ktx2", mimeType: "image/ktx2" });
    assert.deepEqual(r.sampler, json.samplers[0]);
    assert.deepEqual(json, before);
    for (const v of [plan.textureRequests, r, r.image, r.sampler])
      assert.equal(Object.isFrozen(v), true);
    json.textures[0].extensions[ext].source = 0;
    json.images[1].uri = "changed";
    json.samplers[0].wrapS = 10497;
    assert.equal(r.image.uri, "compressed.ktx2");
    assert.equal(r.sampler.wrapS, 33071);
    const borrowed = resource(),
      resolved = plan.resolveTextures((request) => {
        assert.equal(request, r);
        return borrowed;
      });
    assert.equal(resolved.drawables[0].baseColorTexture.view, borrowed.view);
    assert.equal(resolved.accelerationClaim, false);
  });
  test("required extension fails without support before geometry and resolver effects", () => {
    const json = fixture();
    json.extensionsRequired = [ext];
    const before = decodes;
    assert.throws(() => decode(json, [], { resolveTexture: unused }), {
      code: "GLTF_MODEL_UNSUPPORTED",
    });
    assert.equal(decodes, before);
    delete json.textures[0].source;
    assert.equal(prepare(json, [], { basisu: true }).textureRequests[0].imageIndex, 1);
  });
  test("optional extension without a fallback is never replaced with invented pixels", () => {
    const json = fixture();
    delete json.textures[0].source;
    assert.throws(() => prepare(json, []), { code: "GLTF_MODEL_TEXTURE" });
    assert.equal(prepare(json, [], { basisu: true }).textureRequests[0].imageIndex, 1);
  });
  test("unselected compressed source and fallback are not traversed", () => {
    const json = fixture();
    json.images[1] = { uri: 9, bufferView: 99 };
    assert.equal(prepare(json, []).textureRequests[0].imageIndex, 0);
    json.images[1] = { uri: "compressed.ktx2" };
    json.textures[0].source = 999;
    assert.equal(prepare(json, [], { basisu: true }).textureRequests[0].imageIndex, 1);
  });
  test("ordinary PNG/JPEG textures remain valid with BasisU enabled", () => {
    const json = fixture();
    delete json.textures[0].extensions;
    assert.deepEqual(
      prepare(json, [], { basisu: true }).textureRequests,
      prepare(json, []).textureRequests,
    );
  });
  test("embedded KTX2 requires its MIME and snapshots its buffer-view range", () => {
    const json = fixture();
    json.images[1] = { bufferView: 0, mimeType: "image/ktx2" };
    json.buffers = [{ byteLength: 512 }];
    json.bufferViews = [{ buffer: 0, byteOffset: 16, byteLength: 176 }];
    const r = prepare(json, [], { basisu: true }).textureRequests[0];
    assert.deepEqual(r.image, {
      bufferView: 0,
      buffer: 0,
      byteOffset: 16,
      byteLength: 176,
      mimeType: "image/ktx2",
    });
    json.bufferViews[0].byteOffset = 400;
    assert.equal(r.image.byteOffset, 16);
    assert.throws(() => prepare(json, [], { basisu: true }), { code: "GLTF_MODEL_TEXTURE" });
    json.bufferViews[0].byteOffset = 0;
    delete json.images[1].mimeType;
    assert.throws(() => prepare(json, [], { basisu: true }), { code: "GLTF_MODEL_TEXTURE" });
  });
  for (const mimeType of ["image/png", "image/jpeg", "image/webp", "application/octet-stream"])
    test("BasisU does not accept a declared " + mimeType + " source", () => {
      const json = fixture();
      json.images[1].mimeType = mimeType;
      assert.throws(() => decode(json, [], { basisu: true, resolveTexture: unused }), {
        code: "GLTF_MODEL_TEXTURE",
      });
    });
  for (const value of [
    null,
    [],
    {},
    { source: -1 },
    { source: 99 },
    { source: 1.5 },
    { source: 1, extensions: { unknown: {} } },
  ])
    test("invalid selected BasisU descriptor " + JSON.stringify(value), () => {
      const json = fixture();
      json.textures[0].extensions[ext] = value;
      assert.throws(() => decode(json, [], { basisu: true, resolveTexture: unused }), PoseError);
    });
  test("unknown required and texture extensions remain explicit refusals", () => {
    const json = fixture();
    json.extensionsRequired = ["UNKNOWN"];
    assert.throws(() => prepare(json, [], { basisu: true }), { code: "GLTF_MODEL_UNSUPPORTED" });
    delete json.extensionsRequired;
    json.textures[0].extensions.UNKNOWN = {};
    assert.throws(() => prepare(json, [], { basisu: true }), { code: "GLTF_MODEL_UNSUPPORTED" });
  });
  test("distinct samplers, per-map UV transforms and color spaces survive selection", () => {
    const json = fixture();
    json.textures.push({ extensions: { [ext]: { source: 1 } } });
    const m = json.materials[0];
    m.emissiveTexture = {
      index: 0,
      extensions: { KHR_texture_transform: { offset: [0.25, 0.5], texCoord: 1 } },
    };
    m.normalTexture = { index: 1, scale: 0.5 };
    json.materials.push(structuredClone(m));
    const plan = prepare(json, [], { basisu: true });
    assert.deepEqual(
      plan.textureRequests.map((r) => [r.textureIndex, r.imageIndex, r.colorSpace]),
      [
        [0, 1, "srgb"],
        [1, 1, "linear"],
      ],
    );
    let calls = 0;
    const result = plan.resolveTextures(() => {
      calls++;
      return resource();
    });
    assert.equal(calls, 2);
    const a = result.drawables[0],
      b = result.drawables[1];
    assert.equal(a.baseColorTexture, a.emissiveTexture);
    assert.equal(a.baseColorTexture, b.baseColorTexture);
    assert.notEqual(a.normalTexture, a.baseColorTexture);
    assert.equal(a.normalScale, 0.5);
    assert.deepEqual(a.mapCoordinates.emissiveTexture.uvTransform, [1, 0, -0, 1, 0.25, 0.5]);
    assert.equal(a.mapCoordinates.emissiveTexture.texCoords[0], 2);
    assert.deepEqual(a.uvTransform, [1, 0, 0, 1, 0, 0]);
  });
  test("failed resolver never retries fallback; prepared snapshot can be retried explicitly", () => {
    const plan = prepare(fixture(), [], { basisu: true }),
      requests = [],
      failure = new Error("transcode failed");
    assert.throws(
      () =>
        plan.resolveTextures((r) => {
          requests.push(r.imageIndex);
          throw failure;
        }),
      (error) => error === failure,
    );
    plan.resolveTextures((r) => {
      requests.push(r.imageIndex);
      return resource();
    });
    assert.deepEqual(requests, [1, 1]);
    assert.throws(() => plan.resolveTextures(resource), { code: "GLTF_MODEL_PREPARED" });
  });
  test("invalid capability flags fail before any decoding", () => {
    for (const basisu of [null, 1, "true", {}, []]) {
      const before = decodes;
      assert.throws(() => prepare(fixture(), [], { basisu }), { code: "GLTF_MODEL_TEXTURE" });
      assert.equal(decodes, before);
    }
  });
}
