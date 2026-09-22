import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createCpuGltfAnimationModel, decodeGltfAnimationModel } from "./animation_model.mjs";
import { createAnimationModelPicker } from "./animation_model_pick.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { createGltfSceneView } from "./gltf_scene_view.mjs";

// Only GPU-scene construction/submission is replaced. Model/accessor/geometry
// decoding, scene views, CPU pose evaluation, deformation and BVH are real code.
const encoded = (text) => "data:text/javascript;base64," + Buffer.from(text).toString("base64");
const sceneBoundary = encoded(
  "export async function createGpuAnimationScene(device,pose,drawables,options){return device.createScene(pose,drawables,options);}",
);
let gpuSource = readFileSync(new URL("./animation_model_gpu.mjs", import.meta.url), "utf8");
for (const name of [
  "animation_model.mjs",
  "animation_runtime.mjs",
  "animation_scene.mjs",
  "animation_model_pick.mjs",
])
  gpuSource = gpuSource.replace(
    "'./" + name + "'",
    JSON.stringify(
      name === "animation_scene.mjs" ? sceneBoundary : new URL("./" + name, import.meta.url).href,
    ),
  );
const gpuModule = encoded(gpuSource);
const { createGpuGltfAnimationScene, createGpuDecodedAnimationScene } = await import(gpuModule);
const near = (a, b, e = 1e-6) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) < e, `${i}: ${a[i]} != ${b[i]}`);
};
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const ray = (x = 10, z = 5) => ({ origin: [x, 0, z], direction: [0, 0, -1] });
const cameraOptions = { cameraNode: 3, aspectRatio: 1 };
const texture = () => ({ view: {}, sampler: {} });
function fixture({ orthographic = false, textured = false } = {}) {
  const json = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [
      { translation: [10, 0, 0], children: [1, 2, 3] },
      { mesh: 0, skin: 0, weights: [0] },
      {},
      { camera: 0, translation: [0, 0, 5] },
    ],
    skins: [{ joints: [2] }],
    cameras: [
      orthographic
        ? { type: "orthographic", orthographic: { xmag: 2, ymag: 2, znear: 0, zfar: 100 } }
        : { type: "perspective", perspective: { yfov: Math.PI / 2, znear: 0.1, zfar: 100 } },
    ],
    meshes: [{ primitives: [{ attributes: {}, material: 0, targets: [] }] }],
    materials: [{ pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 1 } }],
    buffers: [],
    bufferViews: [],
    accessors: [],
    animations: [],
  };
  const buffers = [];
  function attr(values, type, componentType = 5126) {
    const data = componentType === 5126 ? new Float32Array(values) : new Uint16Array(values),
      buffer = buffers.push(data) - 1;
    json.buffers.push({ byteLength: data.byteLength });
    const bufferView = json.bufferViews.push({ buffer, byteLength: data.byteLength }) - 1;
    return (
      json.accessors.push({
        bufferView,
        type,
        componentType,
        count: values.length / { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[type],
      }) - 1
    );
  }
  const p = json.meshes[0].primitives[0];
  p.attributes.POSITION = attr([-1, -1, 0, 1, -1, 0, 0, 1, 0], "VEC3");
  json.accessors[0].min = [-1, -1, 0];
  json.accessors[0].max = [1, 1, 0];
  p.attributes.NORMAL = attr([0, 0, 1, 0, 0, 1, 0, 0, 1], "VEC3");
  p.attributes.JOINTS_0 = attr(Array(12).fill(0), "VEC4", 5123);
  p.attributes.WEIGHTS_0 = attr([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "VEC4");
  p.attributes.TEXCOORD_0 = attr([0, 0, 1, 0, 0.5, 1], "VEC2");
  p.indices = attr([0, 1, 2], "SCALAR", 5123);
  p.targets = [{ POSITION: attr([2, 0, 0, 2, 0, 0, 2, 0, 0], "VEC3") }];
  const times = attr([0, 1], "SCALAR");
  json.accessors[times].min = [0];
  json.accessors[times].max = [1];
  const animated = [
    { node: 0, path: "translation", type: "VEC3", values: [10, 0, 0, 14, 0, 0] },
    { node: 2, path: "translation", type: "VEC3", values: [0, 0, 0, 0, 0, 2] },
    { node: 1, path: "weights", type: "SCALAR", values: [0, 1] },
    { node: 3, path: "translation", type: "VEC3", values: [0, 0, 5, 2, 0, 5] },
  ];
  json.animations = [
    {
      samplers: animated.map((a) => ({ input: times, output: attr(a.values, a.type) })),
      channels: animated.map((a, sampler) => ({ sampler, target: { node: a.node, path: a.path } })),
    },
  ];
  if (textured) {
    json.textures = [{ source: 0 }];
    json.images = [{ uri: "base.png" }];
    json.materials[0].pbrMetallicRoughness.baseColorTexture = { index: 0 };
  }
  return { json, buffers };
}
function device({ beforeReady = () => {} } = {}) {
  let pose,
    creates = 0,
    updates = 0,
    uploads = 0;
  const frames = [];
  const scene = {
    controller: {},
    draws: [{}],
    deformers: [],
    poseVersion: 0,
    bufferBytes: 128,
    disposed: false,
    failed: false,
    update(time, options) {
      updates++;
      pose.sample(time, options);
      this.upload();
    },
    upload() {
      uploads++;
      this.poseVersion = pose.version;
      for (const d of this.deformers) d.poseVersion = pose.version;
    },
    render(frame) {
      if (this.disposed) throw Error("disposed");
      frames.push(frame);
    },
    async whenIdle() {},
    dispose() {
      this.disposed = true;
      for (const d of this.deformers) d.disposed = true;
    },
  };
  return {
    scene,
    frames,
    get pose() {
      return pose;
    },
    get creates() {
      return creates;
    },
    get updates() {
      return updates;
    },
    get uploads() {
      return uploads;
    },
    async createScene(p, drawables, options) {
      creates++;
      pose = p;
      this.options = options;
      scene.deformers = drawables.map(() => ({ poseVersion: p.version, disposed: false }));
      scene.poseVersion = p.version;
      await beforeReady(p, drawables);
      return scene;
    },
  };
}
function cpu(f = fixture(), picking = true) {
  return createCpuGltfAnimationModel(f.json, f.buffers, { picking, resolveTexture: texture });
}
function gpu(f = fixture(), picking = true, d = device()) {
  return createGpuGltfAnimationScene(d, f.json, f.buffers, {
    picking,
    decode: { resolveTexture: texture },
  });
}

test("CPU factory returns source IDs, UVs and immutable current-pose hits through both query APIs", () => {
  const m = cpu(fixture({ textured: true })),
    h = m.pick([0, 0], cameraOptions, { firstHitOnly: true })[0];
  assert.equal(m.pickingEnabled, true);
  assert.equal(h.distance, 5);
  near(h.point, [10, 0, 0]);
  near(h.uv, [0.5, 0.5]);
  assert.deepEqual(h.source, { node: 1, mesh: 0, primitive: 0, material: 0 });
  assert.deepEqual(m.raycast(ray()), [h]);
  assert.ok(Object.isFrozen(h));
  assert.equal(m.pickingStats.poseVersion, m.pose.version);
  m.dispose();
});
test("morph then skin, parent animation and camera movement select the new geometry, not the rest mesh", () => {
  const m = cpu(),
    old = m.pick([0, 0], cameraOptions)[0];
  m.sample(0.5);
  const h = m.pick([0, 0], cameraOptions)[0];
  near(h.point, [13, 0, 1]);
  assert.equal(h.distance, 4);
  assert.equal(m.raycast(ray()).length, 0);
  assert.equal(old.distance, 5);
  near(old.point, [10, 0, 0]);
  m.sample(1);
  near(m.pick([0, 0], cameraOptions)[0].point, [16, 0, 2]);
  m.reset();
  assert.equal(m.pick([0, 0], cameraOptions)[0].distance, 5);
  m.dispose();
});
test("CPU queries reuse existing deformation without changing versions or advancing animation", () => {
  const m = cpu();
  m.sample(0.5);
  const version = m.pose.version,
    meshVersion = m.deformers[0].version;
  m.pick([0, 0], cameraOptions);
  assert.equal(m.pickingStats.refittedMeshes, 1);
  m.raycast(ray(13));
  assert.equal(m.pickingStats.refittedMeshes, 0);
  assert.equal(m.pose.version, version);
  assert.equal(m.deformers[0].version, meshVersion);
  m.dispose();
});
test("CPU direct sampling and blended root transforms require update before picking", () => {
  const m = cpu(),
    root = identity();
  root[12] = 100;
  root[13] = 200;
  root[14] = 300;
  m.pick([0, 0], cameraOptions);
  const stats = m.pickingStats;
  m.pose.blend([{ clip: 0, time: 1, weight: 0.5 }], { rootMatrix: root });
  assert.throws(() => m.pick([0, 0], cameraOptions), { code: "ANIMATION_PICK_STALE" });
  assert.equal(m.pickingStats, stats);
  m.update();
  const h = m.pick([0, 0], cameraOptions)[0];
  near(h.point, [113, 200, 301]);
  assert.equal(h.distance, 4);
  m.dispose();
});
test("CPU picking metadata is isolated from later exposed drawable and source edits", () => {
  const m = cpu(fixture({ textured: true }));
  m.drawables[0].indices.fill(0);
  m.drawables[0].texCoords.fill(99);
  m.source[0].mesh = 42;
  const h = m.pick([0, 0], cameraOptions)[0];
  near(h.uv, [0.5, 0.5]);
  assert.equal(h.source.mesh, 0);
  m.dispose();
});
test("opt-in preserves default CPU and GPU models and reports an explicit disabled query", async () => {
  const f = fixture(),
    c = cpu(f, false),
    d = device(),
    g = await gpu(f, false, d);
  for (const m of [c, g]) {
    assert.equal(m.pickingEnabled, false);
    assert.equal(m.pickingStats, null);
    assert.throws(() => m.raycast(ray()), { code: "ANIMATION_PICK_DISABLED" });
    assert.throws(() => m.pick([0, 0], cameraOptions), { code: "ANIMATION_PICK_DISABLED" });
  }
  c.sample(0.5);
  g.update(0.5);
  assert.equal(g.render({}), g);
  c.dispose();
  g.dispose();
});
test("GPU query lazily materializes the CPU reference pose without submitting GPU work", async () => {
  const d = device(),
    m = await gpu(fixture(), true, d);
  assert.equal(m.pickingStats, null);
  assert.equal(d.uploads, 0);
  m.update(0.5);
  assert.equal(m.pickingStats, null);
  const uploads = d.uploads,
    version = m.pose.version;
  const h = m.pick([0, 0], cameraOptions)[0];
  near(h.point, [13, 0, 1]);
  assert.equal(h.distance, 4);
  assert.equal(m.pickingStats.refittedMeshes, 1);
  m.raycast(ray(13));
  assert.equal(m.pickingStats.refittedMeshes, 0);
  assert.equal(d.uploads, uploads);
  assert.equal(d.frames.length, 0);
  assert.equal(m.pose.version, version);
  m.update(1);
  near(m.pick([0, 0], cameraOptions)[0].point, [16, 0, 2]);
  assert.equal(m.pickingStats.refittedMeshes, 1);
  m.dispose();
});
test("GPU picking source geometry, topology and UVs are snapshotted before asynchronous scene initialization", async () => {
  const f = fixture({ textured: true }),
    prepared = decodeGltfAnimationModel(f.json, f.buffers, { resolveTexture: texture });
  const d = device({
    beforeReady: async () => {
      await Promise.resolve();
      prepared.drawables[0].geometry.positions.fill(99);
      prepared.drawables[0].geometry.morphTargets[0].positions.fill(99);
      prepared.drawables[0].indices.fill(0);
      prepared.drawables[0].texCoords.fill(99);
      prepared.source[0].mesh = 77;
    },
  });
  const m = await createGpuDecodedAnimationScene(d, prepared, { picking: true });
  m.update(0.5);
  const h = m.pick([0, 0], cameraOptions)[0];
  near(h.point, [13, 0, 1]);
  near(h.uv, [0.5, 0.5]);
  assert.equal(h.source.mesh, 0);
  m.dispose();
});
test("GPU direct pose edits reject stale uploads even on the first query and recover after upload", async () => {
  const d = device(),
    m = await gpu(fixture(), true, d);
  m.pose.sample(1);
  assert.throws(() => m.pick([0, 0], cameraOptions), { code: "ANIMATION_PICK_STALE" });
  assert.equal(m.pickingStats, null);
  assert.equal(d.uploads, 0);
  m.upload();
  assert.equal(m.pick([0, 0], cameraOptions)[0].distance, 3);
  m.dispose();
});
test("orthographic camera picking follows animated position with parallel rays on CPU and GPU models", async () => {
  const f = fixture({ orthographic: true }),
    c = cpu(f),
    g = await gpu(f);
  for (const m of [c, g]) {
    m === c ? m.sample(0.5) : m.update(0.5);
    const h = m.pick([0, 0], { cameraNode: 3 })[0];
    near(h.point, [13, 0, 1]);
    assert.equal(h.distance, 4);
    assert.equal(m.pick([1, 1], { cameraNode: 3 }).length, 0);
    m.dispose();
  }
});
test("raw raycast works without authored cameras and camera failure keeps the model usable", async () => {
  const f = fixture();
  delete f.json.nodes[3].camera;
  const c = cpu(f),
    g = await gpu(f);
  for (const m of [c, g]) {
    assert.throws(() => m.pick([0, 0]), { code: "GLTF_VIEW_CAMERA" });
    assert.equal(m.raycast(ray())[0].distance, 5);
    m.dispose();
  }
});
test("camera, ray and query getters cannot reenter model updates, rendering or disposal", async () => {
  const c = cpu(),
    g = await gpu();
  for (const m of [c, g]) {
    const run = m === c ? () => m.sample(0.5) : () => m.update(0.5),
      code = m === c ? "GLTF_MODEL_REENTRANT" : "GLTF_VIEW_REENTRANT";
    for (const operation of [run, () => m.dispose(), () => m.raycast(ray())]) {
      assert.throws(
        () =>
          m.raycast({
            get origin() {
              operation();
              return [10, 0, 5];
            },
            direction: [0, 0, -1],
          }),
        { code },
      );
      assert.throws(
        () =>
          m.pick([0, 0], {
            get aspectRatio() {
              operation();
              return 1;
            },
          }),
        { code },
      );
      assert.throws(
        () =>
          m.raycast(ray(), {
            get firstHitOnly() {
              operation();
              return true;
            },
          }),
        { code },
      );
    }
    assert.equal(m.pickingStats, null);
    assert.equal(m.pose.disposed, false);
    assert.equal(m.raycast(ray()).length, 1);
    m.dispose();
  }
});
test("direct pose mutation from ray or NDC getters fails without exposing mixed-camera/mesh hits", async () => {
  const c = cpu(),
    g = await gpu();
  for (const m of [c, g]) {
    m.pick([0, 0], cameraOptions);
    const stats = m.pickingStats;
    assert.throws(
      () =>
        m.raycast({
          get origin() {
            m.pose.sample(1);
            return [16, 0, 5];
          },
          direction: [0, 0, -1],
        }),
      { code: "ANIMATION_PICK_CHANGED" },
    );
    assert.equal(m.pickingStats, stats);
    m === c ? m.update() : m.upload();
    near(m.pick([0, 0], cameraOptions)[0].point, [16, 0, 2]);
    assert.throws(
      () =>
        m.pick([0, 0], {
          get aspectRatio() {
            m.pose.reset();
            return 1;
          },
        }),
      { code: "GLTF_VIEW_CHANGED" },
    );
    m === c ? m.update() : m.upload();
    assert.equal(m.pick([0, 0], cameraOptions)[0].distance, 5);
    m.dispose();
  }
});
test("query failures retain the model and successful statistics, and dispose releases picking state", async () => {
  const c = cpu(),
    g = await gpu();
  for (const m of [c, g]) {
    m.raycast(ray());
    const stats = m.pickingStats;
    assert.throws(() => m.raycast({ origin: [0, 0, 0], direction: [0, 0, 0] }), {
      code: "ANIMATION_PICK_RAY",
    });
    assert.equal(m.pickingStats, stats);
    assert.equal(m.failed, false);
    m.dispose();
    assert.equal(m.pickingStats, null);
    assert.throws(() => m.raycast(ray()));
    assert.equal(m.pose.disposed, true);
  }
});
test("normal-only morphs do not require copying normal buffers for GPU selection", async () => {
  const f = fixture();
  f.json.meshes[0].primitives[0].targets[0] = {
    NORMAL: f.json.meshes[0].primitives[0].targets[0].POSITION,
  };
  const d = device(),
    m = await gpu(f, true, d);
  m.update(1);
  const h = m.raycast(ray(14))[0];
  near(h.point, [14, 0, 2]);
  assert.equal(h.distance, 3);
  m.dispose();
});
test("invalid opt-in settings and aggregate source budgets fail before GPU scene creation", async () => {
  for (const picking of [
    null,
    0,
    "yes",
    [],
    { unknown: 1 },
    { maxTriangles: 0 },
    { maxBytes: 0 },
    { maxComponents: 1 },
  ]) {
    const d = device();
    await assert.rejects(gpu(fixture(), picking, d));
    assert.equal(d.creates, 0);
  }
});
test("a query-time BVH budget failure is recoverable for rendering and does not expose partial hits", async () => {
  const d = device(),
    m = await gpu(fixture(), { maxBytes: 1 }, d);
  assert.throws(() => m.raycast(ray()), { code: "ANIMATION_PICK_LIMIT" });
  assert.equal(m.pickingStats, null);
  assert.equal(m.failed, false);
  assert.equal(m.render({}), m);
  assert.equal(d.frames.length, 1);
  assert.throws(() => m.raycast(ray()), { code: "ANIMATION_PICK_LIMIT" });
  m.dispose();
});
test("CPU picking uses the core triangle and memory budgets at construction", () => {
  assert.throws(() => cpu(fixture(), { maxBytes: 1 }), { code: "ANIMATION_PICK_LIMIT" });
  const f = fixture();
  f.json.meshes[0].primitives.push(structuredClone(f.json.meshes[0].primitives[0]));
  assert.throws(() => cpu(f, { maxTriangles: 1 }), { code: "ANIMATION_PICK_LIMIT" });
});
test("source snapshot aggregate budget is enforced across GPU meshes, not separately for each", async () => {
  const f = fixture(),
    single = await gpu(f, { maxComponents: 45 });
  single.dispose();
  f.json.meshes[0].primitives.push(structuredClone(f.json.meshes[0].primitives[0]));
  const d = device();
  await assert.rejects(gpu(f, { maxComponents: 45 }, d), { code: "ANIMATION_PICK_LIMIT" });
  assert.equal(d.creates, 0);
});
test("selected scene membership and explicit draw filtering preserve original node and primitive IDs", async () => {
  const f = fixture();
  f.json.nodes.push({ mesh: 0, skin: 0, weights: [0] });
  f.json.scenes.push({ nodes: [4] });
  const c = createCpuGltfAnimationModel(f.json, f.buffers, { scene: 1, picking: true });
  assert.deepEqual(c.source, [{ node: 4, mesh: 0, primitive: 0, material: 0 }]);
  const h = c.raycast(ray())[0];
  assert.equal(h.node, 4);
  assert.equal(h.source.node, 4);
  assert.deepEqual(c.raycast(ray(), { drawIndices: [] }), []);
  c.dispose();
});
test("GPU terminal draw/completion failure clears picking storage and disposes the owned pose", async () => {
  for (const completion of [false, true]) {
    const d = device(),
      m = await gpu(fixture(), true, d);
    m.pick([0, 0], cameraOptions);
    const fail = () => {
      d.scene.failed = true;
      throw Error("device lost");
    };
    if (completion) {
      d.scene.whenIdle = async () => fail();
      await assert.rejects(m.whenIdle(), /device lost/);
    } else {
      d.scene.render = fail;
      assert.throws(() => m.render({}), /device lost/);
    }
    assert.equal(m.pickingStats, null);
    assert.equal(m.pose.disposed, true);
    assert.throws(() => m.raycast(ray()));
    m.dispose();
  }
});
test("GPU construction failure disposes the newly owned pose", async () => {
  const d = device({
    beforeReady: () => {
      throw Error("allocation refused");
    },
  });
  await assert.rejects(gpu(fixture(), true, d), /allocation refused/);
  assert.equal(d.pose.disposed, true);
});
test("adapter retains no CPU deformer at preparation and lazily reads the actual pose on first query", () => {
  const f = fixture(),
    decoded = decodeGltfAnimationModel(f.json, f.buffers),
    pose = createAnimationPlayer(decoded.definition);
  let reads = 0;
  const borrowed = new Proxy(pose, {
    get(target, key, receiver) {
      if (["jointMatrices", "morphWeights", "worldMatrices"].includes(key)) reads++;
      return Reflect.get(target, key, receiver);
    },
  });
  const picker = createAnimationModelPicker(
    borrowed,
    createGltfSceneView(pose, decoded.sceneView),
    decoded.drawables,
    decoded.source,
    true,
  );
  assert.equal(reads, 0);
  pose.sample(0.5);
  assert.equal(reads, 0);
  near(picker.raycast(ray(13))[0].point, [13, 0, 1]);
  assert.ok(reads > 0);
  picker.dispose();
  assert.equal(pose.disposed, false);
  assert.throws(() => picker.raycast(ray()), { code: "ANIMATION_PICK_DISPOSED" });
  pose.dispose();
});
test("lazy picker cannot accidentally hold or touch native texture resources", async () => {
  const f = fixture(),
    decoded = decodeGltfAnimationModel(f.json, f.buffers);
  decoded.drawables[0].baseColorTexture = {
    get view() {
      assert.fail("No texture access in CPU selection");
    },
  };
  const pose = createAnimationPlayer(decoded.definition),
    picker = createAnimationModelPicker(
      pose,
      createGltfSceneView(pose, decoded.sceneView),
      decoded.drawables,
      decoded.source,
      true,
    );
  assert.equal(picker.raycast(ray())[0].distance, 5);
  picker.dispose();
  pose.dispose();
});

// The owning-loader seam supplies already loaded asset bytes and native texture
// ownership. Unchanged HTTP/image/WebGPU code is not simulated as native evidence.
const assetBoundary = encoded(
  "export class GltfAssetError extends Error {}\nexport async function loadGltfAsset(source,options){return options.fetch(source);}",
);
const textureBoundary = encoded(
  "export class GltfTextureError extends Error {}\nexport async function createGltfTextureResources(device,requests,readImage,options){return device.textureResources(requests,readImage,options);}",
);
let loaderSource = readFileSync(new URL("./gltf_scene_loader.mjs", import.meta.url), "utf8");
for (const [name, url] of [
  ["gltf_asset.mjs", assetBoundary],
  ["gltf_textures.mjs", textureBoundary],
  ["animation_model_gpu.mjs", gpuModule],
  ["animation_model.mjs", new URL("./animation_model.mjs", import.meta.url).href],
])
  loaderSource = loaderSource.replace("'./" + name + "'", JSON.stringify(url));
const { loadGpuGltfAnimationScene } = await import(encoded(loaderSource));
async function loaded({ picking = true } = {}) {
  const f = fixture({ textured: true }),
    d = device();
  let disposed = 0;
  const resources = {
    failed: false,
    textureBytes: 4,
    resolveTexture: texture,
    dispose() {
      disposed++;
    },
  };
  d.textureResources = async () => resources;
  const m = await loadGpuGltfAnimationScene(d, "model.glb", {
    picking,
    assets: {
      fetch: async () => ({
        json: f.json,
        buffers: f.buffers,
        bytesLoaded: 123,
        readImage() {
          assert.fail("test texture boundary");
        },
      }),
    },
  });
  return {
    m,
    d,
    resources,
    get disposed() {
      return disposed;
    },
  };
}
test("owning URL/GLB loader forwards opt-in, hit results and statistics without changing fluent methods", async () => {
  const f = await loaded();
  assert.equal(f.m.pickingEnabled, true);
  assert.equal(f.m.assetBytes, 123);
  assert.equal(f.m.textureBytes, 4);
  assert.equal(f.m.update(0.5), f.m);
  const hits = f.m.pick([0, 0], cameraOptions, { firstHitOnly: true });
  assert.ok(Array.isArray(hits));
  near(hits[0].point, [13, 0, 1]);
  assert.equal(f.m.raycast(ray(13))[0].distance, 4);
  assert.equal(f.m.pickingStats.hitCount, 1);
  assert.equal(f.m.renderCamera({}, cameraOptions), f.m);
  assert.equal(await f.m.whenIdle(), f.m);
  f.m.dispose();
  assert.equal(f.disposed, 1);
  assert.equal(f.m.pickingStats, null);
});
test("owning loader leaves queries disabled unless explicitly enabled", async () => {
  const f = await loaded({ picking: false });
  assert.equal(f.m.pickingEnabled, false);
  assert.throws(() => f.m.pick([0, 0], cameraOptions), { code: "ANIMATION_PICK_DISABLED" });
  assert.equal(f.m.failed, false);
  f.m.dispose();
});
test("owning loader texture-loss query releases pose and picking state, not just texture resources", async () => {
  const f = await loaded();
  f.m.raycast(ray());
  f.resources.failed = true;
  assert.throws(() => f.m.raycast(ray()), /GLTF_TEXTURE_DEVICE_LOST/);
  assert.equal(f.m.pose.disposed, true);
  assert.equal(f.m.pickingStats, null);
  assert.equal(f.disposed, 1);
});
test("owning loader rejected reentrant disposal does not prematurely free native textures", async () => {
  const f = await loaded();
  assert.throws(
    () =>
      f.m.raycast({
        get origin() {
          f.m.dispose();
          return [10, 0, 5];
        },
        direction: [0, 0, -1],
      }),
    { code: "GLTF_VIEW_REENTRANT" },
  );
  assert.equal(f.disposed, 0);
  assert.equal(f.m.pose.disposed, false);
  assert.equal(f.m.raycast(ray())[0].distance, 5);
  f.m.dispose();
  assert.equal(f.disposed, 1);
});
