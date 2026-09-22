import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { decodeGltfAnimationModel } from "./animation_model.mjs";
import { createAnimationModelExporter } from "./animation_model_export.mjs";
import { exportAnimationPoseGLB } from "./animation_pose_export.mjs";

const EXT = "KHR_materials_clearcoat",
  COAT = ["clearcoatTexture", "clearcoatRoughnessTexture", "clearcoatNormalTexture"];
const MAPS = [
  "baseColorTexture",
  "metallicRoughnessTexture",
  "normalTexture",
  "emissiveTexture",
  "occlusionTexture",
  ...COAT,
];
const UV0 = [0, 0, 1, 0, 0, 1],
  UV1 = [0.25, 0.5, 0.75, 0.5, 0.25, 1],
  I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const texture = () => ({ view: {}, sampler: {} }),
  close = (a, b) => {
    assert.equal(a.length, b.length);
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-5, `${v} != ${b[i]}`));
  };
const png = Uint8Array.from(
  Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGUlEQVR4nAXBAQ0AAAzDIJbcv+UeRNJNwgM+/wYAhdCTnAAAAABJRU5ErkJggg==",
    "base64",
  ),
);
const encoded = () => ({
  bytes: png,
  mimeType: "image/png",
  sampler: { wrapS: 33071, wrapT: 33648, minFilter: 9987 },
});
function fixture({ maps = true, all = false } = {}) {
  const json = {
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: {}, material: 0 }] }],
      materials: [
        {
          extensions: { [EXT]: { clearcoatFactor: 0.75, clearcoatRoughnessFactor: 0.25 } },
          alphaMode: "BLEND",
          pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 0.75, 0.4] },
          emissiveFactor: [0.25, 0.5, 1],
        },
      ],
      extensionsUsed: [EXT],
      extensionsRequired: [EXT],
      accessors: [],
      bufferViews: [],
      buffers: [],
    },
    buffers = [];
  const p = json.meshes[0].primitives[0],
    material = json.materials[0],
    coating = material.extensions[EXT];
  for (const [name, type, values] of [
    ["POSITION", "VEC3", [0, 0, 0, 1, 0, 0, 0, 1, 0]],
    ["NORMAL", "VEC3", [0, 0, 1, 0, 0, 1, 0, 0, 1]],
    ["TEXCOORD_0", "VEC2", UV0],
    ["TEXCOORD_1", "VEC2", UV1],
  ]) {
    const data = new Float32Array(values),
      buffer = buffers.push(data) - 1;
    json.buffers.push({ byteLength: data.byteLength });
    const bufferView = json.bufferViews.push({ buffer, byteLength: data.byteLength }) - 1;
    p.attributes[name] =
      json.accessors.push({
        bufferView,
        componentType: 5126,
        type,
        count: 3,
        ...(name === "POSITION" ? { min: [0, 0, 0], max: [1, 1, 0] } : {}),
      }) - 1;
  }
  if (maps) {
    json.images = [{ uri: "coating.png" }];
    json.textures = [{ source: 0 }];
    for (const name of COAT) coating[name] = { index: 0 };
    coating.clearcoatNormalTexture = { index: 0, texCoord: 1, scale: -0.5 };
  }
  if (all) {
    material.pbrMetallicRoughness.baseColorTexture = { index: 0 };
    material.pbrMetallicRoughness.metallicRoughnessTexture = { index: 0 };
    material.normalTexture = { index: 0, scale: 0.25 };
    material.emissiveTexture = { index: 0 };
    material.occlusionTexture = { index: 0, strength: 0.5 };
    material.extensions.KHR_materials_emissive_strength = { emissiveStrength: 8 };
  }
  const decoded = decodeGltfAnimationModel(json, buffers, { resolveTexture: texture });
  const pose = { version: 3, disposed: false, nodeCount: 1, worldMatrices: new Float64Array(I()) };
  const entries = decoded.drawables.map((drawable, i) => ({
    drawable,
    source: decoded.source[i],
    deformer: {
      vertexCount: 3,
      positions: drawable.geometry.positions,
      normals: drawable.geometry.normals,
      worldMatrix: new Float64Array(I()),
      poseVersion: 3,
      disposed: false,
    },
  }));
  return {
    json,
    buffers,
    pose,
    decoded,
    entries,
    drawable: entries[0].drawable,
    deformer: entries[0].deformer,
  };
}
function unpack(buffer) {
  const header = new DataView(buffer);
  assert.equal(header.getUint32(0, true), 0x46546c67);
  assert.equal(header.getUint32(4, true), 2);
  assert.equal(header.getUint32(8, true), buffer.byteLength);
  const size = header.getUint32(12, true);
  assert.equal(size % 4, 0);
  assert.equal(header.getUint32(16, true), 0x4e4f534a);
  assert.equal(header.getUint32(24 + size, true), 0x004e4942);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, size))),
    bin = new Uint8Array(buffer, 28 + size, header.getUint32(20 + size, true));
  assert.ok(bin.length - json.buffers[0].byteLength < 4);
  return { json, bin };
}
const write = (f, options = {}) =>
  exportAnimationPoseGLB(f.pose, f.entries, { resolveTexture: encoded, ...options });
const reload = (r) => decodeGltfAnimationModel(r.json, [r.bin], { resolveTexture: texture });

test("actual GLB export/re-import preserves coating maps, distinct normal scale, alpha and encoded PNG", async () => {
  const f = fixture({ all: true }),
    uses = [],
    r = unpack(
      await write(f, {
        resolveTexture: (_, { colorSpaces }) => {
          uses.push(colorSpaces);
          return encoded();
        },
      }),
    );
  const m = r.json.materials[0],
    c = m.extensions[EXT],
    d = reload(r).drawables[0];
  assert.equal(c.clearcoatFactor, 0.75);
  assert.equal(c.clearcoatRoughnessFactor, 0.25);
  assert.equal(c.clearcoatNormalTexture.scale, -0.5);
  assert.equal(m.normalTexture.scale, 0.25);
  assert.equal(m.alphaMode, "BLEND");
  assert.equal(d.baseColor[3], 0.4);
  assert.equal(d.clearcoatNormalScale, -0.5);
  assert.equal(d.normalScale, 0.25);
  assert.deepEqual(uses, [["srgb"], ["linear"]]);
  assert.equal(r.json.textures.length, 2);
  assert.equal(r.json.images.length, 1);
  const v = r.json.bufferViews[r.json.images[0].bufferView];
  assert.deepEqual(r.bin.slice(v.byteOffset, v.byteOffset + v.byteLength), png);
  assert.equal(c.clearcoatTexture.index, m.pbrMetallicRoughness.metallicRoughnessTexture.index);
  assert.equal(d.clearcoatNormalTexture, d.occlusionTexture);
  assert.notEqual(d.clearcoatTexture, d.baseColorTexture);
  close(d.emissiveFactor, [2, 4, 8]);
  assert.ok(r.json.extensionsRequired.includes(EXT));
});

test("all eight map UV streams preserve independently baked shared and local transforms", async () => {
  const f = fixture({ all: true }),
    d = f.drawable;
  d.uvTransform = [2, 0, 0, 3, 4, 5];
  d.mapCoordinates = {};
  for (const [i, name] of MAPS.entries())
    d.mapCoordinates[name] = {
      texCoords: new Float64Array(UV1),
      uvTransform: [i + 1, 0, 0, 1, 0.25, 0.5],
    };
  const r = unpack(await write(f)),
    reloaded = reload(r).drawables[0],
    p = r.json.meshes[0].primitives[0];
  assert.equal(Object.keys(p.attributes).filter((k) => k.startsWith("TEXCOORD_")).length, 8);
  const c = r.json.materials[0].extensions[EXT];
  assert.equal(c.clearcoatTexture.texCoord, 5);
  assert.equal(c.clearcoatNormalTexture.texCoord, 7);
  for (const [i, name] of MAPS.entries()) {
    const expected = [];
    for (let v = 0; v < 3; v++)
      expected.push(2 * ((i + 1) * UV1[v * 2] + 0.25) + 4, 3 * (UV1[v * 2 + 1] + 0.5) + 5);
    close(reloaded.mapCoordinates[name].texCoords, expected);
  }
});

for (const factor of [0, 0.5, 1])
  test(`factor-only clearcoat ${factor} preserves empty/default roughness without images`, async () => {
    const f = fixture({ maps: false });
    f.drawable.clearcoatFactor = factor;
    delete f.drawable.clearcoatRoughnessFactor;
    const r = unpack(await write(f, { resolveTexture: () => assert.fail("no texture") }));
    assert.deepEqual(r.json.materials[0].extensions[EXT], {
      clearcoatFactor: factor,
      clearcoatRoughnessFactor: 0,
    });
    assert.equal(r.json.images, undefined);
    assert.equal(reload(r).drawables[0].clearcoatFactor, factor);
  });

test("coating, HDR emission, other unlit materials and punctual-light declarations coexist", async () => {
  const f = fixture({ all: true }),
    second = {
      drawable: { shading: "unlit" },
      source: { node: 0, mesh: 1, primitive: 0, material: 1 },
      deformer: f.deformer,
    };
  const sceneView = {
    format: "f3d-gltf-scene-view-v1",
    nodeCount: 1,
    cameras: [],
    lights: [{ node: 0, light: 0, type: "point", color: [1, 1, 1], intensity: 2 }],
  };
  for (const entries of [
    [...f.entries, second],
    [second, ...f.entries],
  ]) {
    const r = unpack(
      await exportAnimationPoseGLB(f.pose, entries, { resolveTexture: encoded, sceneView }),
    );
    assert.deepEqual(
      new Set(r.json.extensionsRequired),
      new Set([
        EXT,
        "KHR_materials_emissive_strength",
        "KHR_materials_unlit",
        "KHR_lights_punctual",
      ]),
    );
    assert.equal(r.json.extensionsRequired.length, 4);
    assert.equal(r.json.extensionsUsed.length, 4);
    const coated = r.json.materials.find((m) => m.extensions?.[EXT]);
    assert.equal(coated.extensions.KHR_materials_emissive_strength.emissiveStrength, 8);
    assert.equal(reload(r).sceneView.lights.length, 1);
  }
});

test("uncoated materials do not acquire a clearcoat extension during export", async () => {
  const f = fixture({ maps: false });
  delete f.drawable.clearcoatFactor;
  delete f.drawable.clearcoatRoughnessFactor;
  const r = unpack(await write(f));
  assert.equal(r.json.materials[0].extensions, undefined);
  assert.equal(r.json.extensionsRequired, undefined);
});

for (const [name, change] of [
  [
    "negative factor",
    (d) => {
      d.clearcoatFactor = -1;
    },
  ],
  [
    "excess roughness",
    (d) => {
      d.clearcoatRoughnessFactor = 2;
    },
  ],
  [
    "null factor",
    (d) => {
      d.clearcoatFactor = null;
    },
  ],
  [
    "infinite scale",
    (d) => {
      d.clearcoatNormalScale = Infinity;
    },
  ],
  [
    "overflow scale",
    (d) => {
      d.clearcoatNormalScale = 1e100;
    },
  ],
  [
    "null scale",
    (d) => {
      d.clearcoatNormalScale = null;
    },
  ],
  [
    "scale without map",
    (d) => {
      delete d.clearcoatNormalTexture;
    },
  ],
  [
    "bad UV storage",
    (d) => {
      d.mapCoordinates.clearcoatNormalTexture.texCoords = new Float32Array(2);
    },
  ],
  [
    "invalid descriptor",
    (d) => {
      d.clearcoatTexture = { view: {} };
    },
  ],
])
  test(`invalid export ${name} fails before encoded texture resolution`, async () => {
    const f = fixture();
    change(f.drawable);
    let calls = 0;
    await assert.rejects(
      write(f, {
        resolveTexture: () => {
          calls++;
          return encoded();
        },
      }),
    );
    assert.equal(calls, 0);
  });

test("unlit clearcoat cannot be silently dropped even without maps", async () => {
  const f = fixture({ maps: false });
  f.drawable.shading = "unlit";
  await assert.rejects(write(f), { code: "ANIMATION_EXPORT_UNSUPPORTED" });
});

test("a bad later coating prevents all earlier texture callbacks", async () => {
  const f = fixture();
  f.entries.push({ ...f.entries[0], drawable: { ...f.drawable, clearcoatFactor: 2 } });
  let calls = 0;
  await assert.rejects(
    write(f, {
      resolveTexture: () => {
        calls++;
        return encoded();
      },
    }),
    { code: "ANIMATION_EXPORT_VALUE" },
  );
  assert.equal(calls, 0);
});

test("pending export is a complete coating/UV snapshot independent of later disposal and mutation", async () => {
  const f = fixture();
  let finish;
  const pending = write(f, {
    resolveTexture: () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  });
  assert.equal(typeof finish, "function");
  f.drawable.clearcoatFactor = 99;
  f.drawable.clearcoatNormalScale = 99;
  f.drawable.mapCoordinates.clearcoatNormalTexture.texCoords.fill(99);
  f.deformer.positions.fill(99);
  f.pose.version++;
  f.pose.disposed = true;
  f.deformer.disposed = true;
  finish(encoded());
  const r = unpack(await pending),
    d = reload(r).drawables[0];
  assert.equal(d.clearcoatFactor, 0.75);
  assert.equal(d.clearcoatNormalScale, -0.5);
  close(d.mapCoordinates.clearcoatNormalTexture.texCoords, UV1);
  close(d.geometry.positions, [0, 0, 0, 1, 0, 0, 0, 1, 0]);
});

test("cancellation during coating image resolution does not dispose the model or leave an unobserved rejection", async () => {
  const f = fixture(),
    abort = new AbortController();
  let reject;
  const pending = write(f, {
    signal: abort.signal,
    resolveTexture: () =>
      new Promise((_, r) => {
        reject = r;
      }),
  });
  abort.abort();
  await assert.rejects(pending, { name: "AbortError" });
  reject(Error("late image failure"));
  await new Promise((r) => setImmediate(r));
  assert.equal(f.pose.disposed, false);
  assert.equal(f.deformer.disposed, false);
});

test("final GLB budget includes extension metadata, all coating UVs and encoded images", async () => {
  const f = fixture({ all: true }),
    bytes = await write(f);
  await assert.rejects(write(f, { maxBytes: bytes.byteLength - 1 }), {
    code: "ANIMATION_EXPORT_LIMIT",
  });
  assert.deepEqual(
    new Uint8Array(await write(f, { maxBytes: bytes.byteLength })),
    new Uint8Array(bytes),
  );
});

test("the model exporter reuses published CPU geometry while preserving all coating maps", async () => {
  const f = fixture(),
    model = createAnimationModelExporter(f.pose, f.decoded.drawables, f.decoded.source, true, [
      f.deformer,
    ]);
  const r = unpack(await model.exportPoseGLB({ resolveTexture: encoded }));
  assert.equal(reload(r).drawables[0].clearcoatNormalScale, -0.5);
  model.dispose();
  assert.equal(f.pose.disposed, false);
  assert.equal(f.deformer.disposed, false);
});

// For the GPU-model retention branch, replace only CPU deformation with explicit
// supplied output. This exercises the real source/material snapshot and writer,
// not skinning arithmetic, GPU readback or shader execution.
const encode = (s) => "data:text/javascript;base64," + Buffer.from(s).toString("base64");
const exporterCode = readFileSync(new URL("./animation_model_export.mjs", import.meta.url), "utf8")
  .replace(
    "'./animation_deformer.mjs'",
    JSON.stringify(
      encode(
        "export function createAnimationDeformer(pose,geometry){return pose.makeDeformer(geometry);}",
      ),
    ),
  )
  .replaceAll(
    "'./animation_pose_export.mjs'",
    JSON.stringify(new URL("./animation_pose_export.mjs", import.meta.url).href),
  );
const { createAnimationModelExporter: createRetained } = await import(encode(exporterCode));

test("retained GPU-model export snapshots coating descriptors and independent UVs before lazy deformation", async () => {
  const f = fixture(),
    original = f.drawable.clearcoatTexture.view;
  let made = 0,
    disposed = 0;
  f.pose.makeDeformer = (g) => {
    made++;
    return {
      vertexCount: 3,
      positions: g.positions,
      normals: g.normals,
      worldMatrix: new Float64Array(I()),
      poseVersion: 3,
      disposed: false,
      dispose() {
        disposed++;
      },
    };
  };
  f.drawable.clearcoatTexture = { ...f.drawable.clearcoatTexture };
  const model = createRetained(f.pose, f.decoded.drawables, f.decoded.source, true);
  f.drawable.clearcoatTexture.view = {};
  f.drawable.clearcoatFactor = 99;
  f.drawable.mapCoordinates.clearcoatNormalTexture.texCoords.fill(99);
  assert.equal(made, 0);
  const r = unpack(
    await model.exportPoseGLB({
      resolveTexture: (r) => {
        assert.equal(r.view, original);
        return encoded();
      },
    }),
  );
  const d = reload(r).drawables[0];
  assert.equal(d.clearcoatFactor, 0.75);
  close(d.mapCoordinates.clearcoatNormalTexture.texCoords, UV1);
  assert.equal(made, 1);
  model.dispose();
  assert.equal(disposed, 1);
  assert.equal(f.pose.disposed, false);
});

test("retained component limits include all three coating-coordinate snapshots", () => {
  const f = fixture();
  // 18 geometry + 3 indices + 4 color + 3 emission + 6 UV + 6 shared transform
  // + 3 coating maps * (6 UV + 6 local transform) = 76 copied components.
  assert.throws(
    () => createRetained(f.pose, f.decoded.drawables, f.decoded.source, { maxComponents: 75 }),
    { code: "ANIMATION_EXPORT_LIMIT" },
  );
  const m = createRetained(f.pose, f.decoded.drawables, f.decoded.source, { maxComponents: 76 });
  m.dispose();
});
