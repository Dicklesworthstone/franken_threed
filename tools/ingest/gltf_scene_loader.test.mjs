import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createAnimationDeformer } from "./animation_deformer.mjs";
import { createCpuGltfAnimationModel, prepareGltfAnimationModel } from "./animation_model.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { imageBytes, pngBase64 } from "./fixtures/animation/image_fixture.mjs";
import { loadGltfAsset } from "./gltf_asset.mjs";
import { gltfImageDimensions } from "./gltf_textures.mjs";

// Replace ONLY the GPU-scene boundary. Asset I/O, model/texture preparation,
// GPU resource orchestration, pose construction and CPU deformation are real.
// The boundary below is not evidence of native image/shader/pixel execution.
const encoded = (s) => "data:text/javascript;base64," + Buffer.from(s).toString("base64");
const boundary = encoded(
  "export async function createGpuAnimationScene(device,pose,drawables,options){return device.createScene(pose,drawables,options);}",
);
let modelCode = readFileSync(new URL("./animation_model_gpu.mjs", import.meta.url), "utf8");
for (const name of ["animation_model.mjs", "animation_runtime.mjs", "animation_scene.mjs"])
  modelCode = modelCode.replace(
    "'./" + name + "'",
    JSON.stringify(
      name === "animation_scene.mjs" ? boundary : new URL("./" + name, import.meta.url).href,
    ),
  );
const modelModule = encoded(modelCode);
let loaderCode = readFileSync(new URL("./gltf_scene_loader.mjs", import.meta.url), "utf8");
for (const name of [
  "gltf_asset.mjs",
  "animation_model.mjs",
  "gltf_textures.mjs",
  "animation_model_gpu.mjs",
])
  loaderCode = loaderCode.replace(
    "'./" + name + "'",
    JSON.stringify(
      name === "animation_model_gpu.mjs" ? modelModule : new URL("./" + name, import.meta.url).href,
    ),
  );
const { loadGpuGltfAnimationScene } = await import(encoded(loaderCode));
const { createGpuDecodedAnimationScene } = await import(modelModule);
const utf8 = (value) => new TextEncoder().encode(JSON.stringify(value));
const png = imageBytes(pngBase64);
function fixture({ textured = true, embedded = false } = {}) {
  const json = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    buffers: [],
    bufferViews: [],
    accessors: [],
    meshes: [{ primitives: [{ attributes: {}, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.6, 1] } }],
  };
  const chunks = [];
  let length = 0;
  function attr(values, type) {
    const data = new Uint8Array(new Float32Array(values).buffer),
      bufferView = json.bufferViews.length;
    json.bufferViews.push({ buffer: 0, byteOffset: length, byteLength: data.length });
    length += data.length;
    chunks.push(data);
    return (
      json.accessors.push({
        bufferView,
        componentType: 5126,
        type,
        count: values.length / { SCALAR: 1, VEC2: 2, VEC3: 3 }[type],
      }) - 1
    );
  }
  const p = json.meshes[0].primitives[0];
  p.attributes.POSITION = attr([0, 0, 0, 1, 0, 0, 0, 1, 0], "VEC3");
  json.accessors[0].min = [0, 0, 0];
  json.accessors[0].max = [1, 1, 0];
  p.attributes.TEXCOORD_0 = attr([0, 0, 1, 0, 0, 1], "VEC2");
  const time = attr([0, 1], "SCALAR");
  json.accessors[time].min = [0];
  json.accessors[time].max = [1];
  const translation = attr([0, 0, 0, 4, 2, 0], "VEC3");
  json.animations = [
    {
      samplers: [{ input: time, output: translation }],
      channels: [{ sampler: 0, target: { node: 0, path: "translation" } }],
    },
  ];
  if (textured) {
    json.textures = [{ source: 0, sampler: 0 }];
    json.samplers = [{ minFilter: 9729, magFilter: 9728, wrapS: 33071 }];
    json.materials[0].pbrMetallicRoughness.baseColorTexture = { index: 0 };
    if (embedded) {
      const bufferView =
        json.bufferViews.push({ buffer: 0, byteOffset: length, byteLength: png.length }) - 1;
      chunks.push(png);
      length += png.length;
      json.images = [{ bufferView, mimeType: "image/png" }];
    } else json.images = [{ uri: "color.png" }, { uri: "https://unreachable.invalid/unused.png" }];
  }
  const bytes = new Uint8Array(length);
  let at = 0;
  for (const part of chunks) {
    bytes.set(part, at);
    at += part.length;
  }
  json.buffers = [{ byteLength: length, uri: "mesh.bin" }];
  return { json, bytes, p };
}
function glb(f) {
  const json = structuredClone(f.json);
  delete json.buffers[0].uri;
  const text = utf8(json),
    j = Math.ceil(text.length / 4) * 4,
    b = Math.ceil(f.bytes.length / 4) * 4,
    out = new Uint8Array(28 + j + b),
    v = new DataView(out.buffer);
  v.setUint32(0, 0x46546c67, true);
  v.setUint32(4, 2, true);
  v.setUint32(8, out.length, true);
  v.setUint32(12, j, true);
  v.setUint32(16, 0x4e4f534a, true);
  out.fill(32, 20, 20 + j);
  out.set(text, 20);
  v.setUint32(20 + j, b, true);
  v.setUint32(24 + j, 0x004e4942, true);
  out.set(f.bytes, 28 + j);
  return out;
}
function environment(f) {
  const requests = [],
    allocations = [],
    bitmaps = [],
    scopes = [],
    events = [];
  let lost, pose, drawables, settings;
  const nativeScene = {
    controller: {},
    draws: [{}],
    deformers: [],
    bufferBytes: 123,
    disposed: false,
    failed: false,
    update(dt) {
      pose.sample(dt);
      for (const d of this.deformers) d.update();
    },
    upload() {
      for (const d of this.deformers) d.update();
    },
    render(frame) {
      events.push(["render", frame]);
    },
    async whenIdle() {},
    dispose() {
      this.disposed = true;
      for (const d of this.deformers) d.dispose();
    },
  };
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise((resolve) => {
      lost = resolve;
    }),
    destroy() {
      assert.fail("Borrowed device must never be destroyed");
    },
    pushErrorScope(s) {
      scopes.push(s);
    },
    popErrorScope() {
      assert.ok(scopes.pop());
      return Promise.resolve(null);
    },
    createTexture(descriptor) {
      const t = {
        descriptor,
        destroyed: 0,
        destroy() {
          this.destroyed++;
        },
        createView(options) {
          return { texture: this, options };
        },
      };
      allocations.push(t);
      events.push("texture");
      return t;
    },
    createSampler(descriptor) {
      return { descriptor };
    },
    queue: {
      copyExternalImageToTexture() {
        events.push("copy");
      },
      async onSubmittedWorkDone() {
        events.push("idle");
      },
    },
    async createScene(p, d, s) {
      pose = p;
      drawables = d;
      settings = s;
      events.push("scene");
      nativeScene.deformers = d.map((x) => createAnimationDeformer(p, x.geometry));
      return nativeScene;
    },
  };
  const options = {
    assets: {
      fetch: async (url, init) => {
        requests.push({ url, init });
        events.push(url);
        const data = new Map([
          ["https://test.invalid/model.gltf", utf8(f.json)],
          ["https://test.invalid/mesh.bin", f.bytes],
          ["https://test.invalid/color.png", png],
        ]).get(url);
        return new Response(data ?? null, { status: data ? 200 : 404 });
      },
    },
    textures: {
      createImageBitmap: async (blob) => {
        events.push("decode");
        assert.equal(scopes.length, 0);
        const dimensions = gltfImageDimensions(new Uint8Array(await blob.arrayBuffer()), blob.type);
        const b = {
          ...dimensions,
          closed: 0,
          close() {
            this.closed++;
          },
        };
        bitmaps.push(b);
        return b;
      },
    },
    scene: { maxBytes: 4096 },
  };
  return {
    device,
    options,
    requests,
    allocations,
    bitmaps,
    events,
    nativeScene,
    lose: (info) => lost(info),
    get pose() {
      return pose;
    },
    get drawables() {
      return drawables;
    },
    get settings() {
      return settings;
    },
  };
}

test("preparation snapshots geometry/materials once and publishes only frozen requests", () => {
  const f = fixture(),
    original = f.bytes.slice();
  let calls = 0;
  const plan = prepareGltfAnimationModel(f.json, (i) => {
    assert.equal(i, 0);
    calls++;
    return f.bytes;
  });
  assert.equal(calls, 1);
  assert.ok(Object.isFrozen(plan.textureRequests));
  assert.equal(plan.textureRequests.length, 1);
  assert.equal(plan.drawables, undefined);
  assert.ok(Object.isFrozen(plan.textureRequests[0].sampler));
  f.bytes.fill(0);
  f.json.materials[0].pbrMetallicRoughness.baseColorFactor.fill(99);
  assert.throws(() => plan.resolveTextures(), { code: "GLTF_MODEL_TEXTURE" });
  assert.throws(
    () =>
      plan.resolveTextures(() => {
        throw Error("not ready");
      }),
    /not ready/,
  );
  const prepared = plan.resolveTextures(() => ({ view: {}, sampler: {} }));
  assert.equal(calls, 1);
  assert.deepEqual(prepared.drawables[0].baseColor, [0.2, 0.4, 0.6, 1]);
  const p = createAnimationPlayer(prepared.definition),
    d = createAnimationDeformer(p, prepared.drawables[0].geometry);
  p.sample(0.5);
  d.update();
  assert.deepEqual([...d.positions], [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.equal(d.worldMatrix[12], 2);
  assert.equal(d.worldMatrix[13], 1);
  assert.throws(() => plan.resolveTextures(() => ({ view: {}, sampler: {} })), {
    code: "GLTF_MODEL_PREPARED",
  });
  d.dispose();
  p.dispose();
  assert.ok(original.some((v) => v !== 0));
});
test("prepared resolution rejects reentry and asynchronous placeholders, then permits retry", () => {
  const f = fixture(),
    plan = prepareGltfAnimationModel(f.json, [f.bytes]);
  assert.throws(
    () => plan.resolveTextures(() => plan.resolveTextures(() => ({ view: {}, sampler: {} }))),
    { code: "GLTF_MODEL_REENTRANT" },
  );
  assert.throws(() => plan.resolveTextures(async () => ({ view: {}, sampler: {} })), {
    code: "GLTF_MODEL_TEXTURE",
  });
  assert.equal(plan.resolveTextures(() => ({ view: {}, sampler: {} })).drawables.length, 1);
});
test("URL -> image upload -> scene receives real geometry, texture resources and sampled pose", async () => {
  const f = fixture(),
    g = environment(f),
    m = await loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options);
  assert.deepEqual(
    g.requests.map((x) => x.url),
    [
      "https://test.invalid/model.gltf",
      "https://test.invalid/mesh.bin",
      "https://test.invalid/color.png",
    ],
  );
  assert.ok(g.events.indexOf("idle") < g.events.indexOf("scene"));
  assert.equal(g.bitmaps[0].closed, 1);
  assert.equal(g.allocations.length, 1);
  assert.equal(g.drawables[0].baseColorTexture.view.texture, g.allocations[0]);
  assert.deepEqual(g.settings, { maxBytes: 4096 });
  assert.equal(g.drawables[0].baseColorTexture.sampler.descriptor.magFilter, "nearest");
  assert.equal(m.update(0.5), m);
  assert.equal(m.deformers[0].worldMatrix[12], 2);
  assert.equal(m.deformers[0].worldMatrix[13], 1);
  const frame = { colorView: {}, viewProjection: [] };
  assert.equal(m.render(frame), m);
  assert.equal(m.upload(), m);
  assert.equal(await m.whenIdle(), m);
  assert.deepEqual(m.source, [{ node: 0, mesh: 0, primitive: 0, material: 0 }]);
  assert.equal(m.assetBytes, utf8(f.json).length + f.bytes.length + png.length);
  assert.equal(m.textureBytes, 16);
  assert.equal(m.bufferBytes, 123);
  m.dispose();
  m.dispose();
  assert.equal(m.disposed, true);
  assert.equal(m.pose.disposed, true);
  assert.equal(g.allocations[0].destroyed, 1);
});
test("GLB containing geometry and embedded image needs no network", async () => {
  const f = fixture({ embedded: true }),
    g = environment(f);
  g.options.assets.fetch = () => assert.fail("No external resources");
  const m = await loadGpuGltfAnimationScene(g.device, glb(f), g.options);
  assert.equal(m.textureBytes, 16);
  m.update(1);
  assert.equal(m.deformers[0].worldMatrix[12], 4);
  m.dispose();
});
test("untextured models need no ImageBitmap implementation or texture-capable device", async () => {
  const f = fixture({ textured: false }),
    g = environment(f);
  g.options.textures.createImageBitmap = null;
  delete g.device.queue;
  delete g.device.createTexture;
  const m = await loadGpuGltfAnimationScene(g.device, glb(f), g.options);
  assert.equal(m.textureBytes, 0);
  assert.equal(g.bitmaps.length, 0);
  m.dispose();
});
test("byte loader also feeds the unchanged real CPU factory", async () => {
  const f = fixture({ textured: false }),
    asset = await loadGltfAsset(glb(f));
  const m = createCpuGltfAnimationModel(asset.json, asset.buffers);
  m.sample(0.75);
  assert.equal(m.deformers[0].worldMatrix[12], 3);
  assert.equal(m.deformers[0].worldMatrix[13], 1.5);
  m.dispose();
});
for (const [name, mutate] of [
  [
    "material",
    (f) => {
      f.json.materials[0].occlusionTexture = { index: 0 };
    },
  ],
  [
    "geometry",
    (f) => {
      delete f.p.attributes.POSITION;
    },
  ],
  [
    "UV route",
    (f) => {
      f.json.materials[0].pbrMetallicRoughness.baseColorTexture.texCoord = 1;
    },
  ],
])
  test(`invalid ${name} is refused before image fetch, native decode or GPU work`, async () => {
    const f = fixture();
    mutate(f);
    const g = environment(f);
    await assert.rejects(
      loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options),
    );
    assert.equal(g.requests.length, 2);
    assert.equal(g.bitmaps.length, 0);
    assert.equal(g.allocations.length, 0);
    assert.equal(g.pose, undefined);
  });
test("native scene initialization failure releases uploaded textures and newly owned pose", async () => {
  const f = fixture(),
    g = environment(f);
  let pose;
  g.device.createScene = async (p) => {
    pose = p;
    throw Error("initialization failed");
  };
  await assert.rejects(
    loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options),
    /initialization failed/,
  );
  assert.equal(pose.disposed, true);
  assert.equal(g.allocations[0].destroyed, 1);
  assert.equal(g.bitmaps[0].closed, 1);
});
test("image upload failure prevents scene creation and releases its resources", async () => {
  const f = fixture(),
    g = environment(f);
  g.device.queue.copyExternalImageToTexture = () => {
    throw Error("upload failed");
  };
  await assert.rejects(
    loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options),
    /upload failed/,
  );
  assert.equal(g.pose, undefined);
  assert.equal(g.allocations[0].destroyed, 1);
  assert.equal(g.bitmaps[0].closed, 1);
});
test("recoverable render errors retain textures; terminal completion errors release the group", async () => {
  const f = fixture(),
    g = environment(f),
    m = await loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options);
  g.nativeScene.render = () => {
    throw Error("bad frame");
  };
  assert.throws(() => m.render({}), /bad frame/);
  assert.equal(m.failed, false);
  assert.equal(m.pose.disposed, false);
  assert.equal(g.allocations[0].destroyed, 0);
  g.nativeScene.whenIdle = async () => {
    g.nativeScene.failed = true;
    throw Error("terminal GPU error");
  };
  await assert.rejects(m.whenIdle(), /terminal GPU error/);
  assert.equal(m.pose.disposed, true);
  assert.equal(g.allocations[0].destroyed, 1);
  assert.equal(m.failed, true);
});
test("device loss terminates texture use and disposes the model on its next operation", async () => {
  const f = fixture(),
    g = environment(f),
    m = await loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options);
  g.lose({ message: "lost" });
  await Promise.resolve();
  assert.equal(m.failed, true);
  assert.equal(g.allocations[0].destroyed, 1);
  assert.throws(() => m.update(0.5), { code: "GLTF_TEXTURE_DEVICE_LOST" });
  assert.equal(m.pose.disposed, true);
  m.dispose();
});
test("reentrant disposal rejected by the scene does not release still-live textures", async () => {
  const f = fixture(),
    g = environment(f),
    m = await loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", g.options),
    dispose = g.nativeScene.dispose;
  g.nativeScene.dispose = () => {
    throw Error("busy");
  };
  assert.throws(() => m.dispose(), /busy/);
  assert.equal(g.allocations[0].destroyed, 0);
  assert.equal(m.pose.disposed, false);
  g.nativeScene.dispose = dispose;
  m.dispose();
  assert.equal(g.allocations[0].destroyed, 1);
});
test("abort during scene initialization destroys the late scene, textures and pose", async () => {
  const f = fixture(),
    g = environment(f),
    controller = new AbortController(),
    original = g.device.createScene;
  let finish;
  g.device.createScene = async (...args) => {
    const scene = await original(...args);
    await new Promise((resolve) => {
      finish = resolve;
    });
    return scene;
  };
  const pending = loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", {
    ...g.options,
    signal: controller.signal,
  });
  while (!finish) await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  finish();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(g.pose.disposed, true);
  assert.equal(g.nativeScene.disposed, true);
  assert.equal(g.allocations[0].destroyed, 1);
});
test("conflicting signals and a custom borrowed resolver are refused before I/O", async () => {
  const f = fixture(),
    g = environment(f),
    a = new AbortController(),
    b = new AbortController();
  for (const extra of [
    { decode: { resolveTexture: () => ({}) } },
    { signal: a.signal, assets: { signal: b.signal } },
  ]) {
    await assert.rejects(
      loadGpuGltfAnimationScene(g.device, "https://test.invalid/model.gltf", {
        ...g.options,
        ...extra,
      }),
      { code: "GLTF_MODEL_LOAD_OPTIONS" },
    );
  }
  assert.equal(g.requests.length, 0);
});
test("decoded-model GPU entry owns only its pose and keeps borrowed texture resources", async () => {
  const f = fixture(),
    g = environment(f),
    texture = { view: {}, sampler: {} },
    prepared = prepareGltfAnimationModel(f.json, [f.bytes]).resolveTextures(() => texture);
  const m = await createGpuDecodedAnimationScene(g.device, prepared, { maxBytes: 1024 });
  assert.equal(g.drawables[0].baseColorTexture.view, texture.view);
  m.dispose();
  assert.equal(m.pose.disposed, true);
  assert.deepEqual(texture, { view: {}, sampler: {} });
});
