import assert from "node:assert/strict";
import test from "node:test";
import { decodeGltfAnimation } from "./animation_gltf.mjs";
import { AnimationPoseError, createAnimationPlayer } from "./animation_runtime.mjs";
import { animationFixture } from "./fixtures/animation/gltf_fixture.mjs";

const close = (a, b, e = 1e-6) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) < e, `${i}: ${a[i]} != ${b[i]}`);
};

test("decodes strided translations, sparse morph weights and joint-order inverse binds", () => {
  const { model, bytes } = animationFixture();
  let calls = 0;
  const definition = decodeGltfAnimation(model, (index) => {
    assert.equal(index, 0);
    calls++;
    return bytes;
  });
  assert.equal(calls, 1);
  assert.deepEqual(
    definition.nodes.map((n) => n.parent),
    [-1, -1, 1],
  );
  const p = createAnimationPlayer(definition);
  p.sample(1);
  close(p.translations.subarray(3, 6), [11, 1, 0]);
  close(p.morphWeights, [0.5, 0.5]);
  close(p.jointMatrices.subarray(12, 15), [-4, 1, 0]);
  close(p.jointMatrices.subarray(28, 31), [-4, 1, 0]);
  assert.equal(p.clips.length, 2);
  p.sample(1, { clip: 1 });
  close(p.rotations.subarray(8, 12), [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  close(p.translations.subarray(3, 6), [10, 0, 0]);
  close(p.morphWeights, [0.25, 0.75]);
});

test("unreferenced geometry, texture and buffers need not be loaded for pose extraction", () => {
  const { model, bytes } = animationFixture();
  model.buffers.push({ uri: "https://unavailable.invalid/geometry.bin", byteLength: 100 });
  model.images = [{ uri: "https://unavailable.invalid/image.png" }];
  model.accessors.push({ bufferView: 999, componentType: 5126, count: 999999999, type: "VEC3" });
  const data = decodeGltfAnimation(model, [bytes]);
  assert.equal(createAnimationPlayer(data).nodeCount, 3);
});

test("supports normalized byte/short rotations and morph streams", () => {
  const { model, bytes, indices } = animationFixture();
  const rotation = model.accessors[indices.rotations],
    view = model.bufferViews[rotation.bufferView];
  rotation.componentType = 5120;
  rotation.normalized = true;
  view.byteLength = 8;
  bytes.set(new Uint8Array(new Int8Array([0, 0, 0, 127, 0, 0, 90, 90]).buffer), view.byteOffset);
  const p = createAnimationPlayer(decodeGltfAnimation(model, [bytes]));
  p.sample(2, { clip: 1 });
  close(p.rotations.subarray(8, 12), [0, 0, 90 / 127, 90 / 127]);
  const d = animationFixture(),
    weights = d.model.accessors[d.indices.weights],
    sv = d.model.bufferViews[d.indices.sparseValues];
  weights.componentType = 5122;
  weights.normalized = true;
  sv.byteLength = 4;
  new DataView(d.bytes.buffer).setInt16(sv.byteOffset, -32768, true);
  new DataView(d.bytes.buffer).setInt16(sv.byteOffset + 2, 32767, true);
  const q = createAnimationPlayer(decodeGltfAnimation(d.model, [d.bytes]));
  q.sample(1);
  close(q.morphWeights, [0.5, -0.5]);
});

test("source channels with no target node are ignored and reported, not fabricated", () => {
  const { model, bytes } = animationFixture();
  model.animations[0].channels.push({ sampler: 999, target: { path: "translation" } });
  const d = decodeGltfAnimation(model, [bytes]);
  assert.deepEqual(d.ignoredChannels, [{ animation: 0, channel: 2, reason: "NO_TARGET_NODE" }]);
  assert.equal(d.clips[0].channels.length, 2);
});

test("decoder snapshots source metadata and sparse arrays without mutating the model/buffers", () => {
  const { model, bytes } = animationFixture(),
    original = structuredClone(model),
    before = bytes.slice();
  const d = decodeGltfAnimation(model, [bytes]);
  assert.deepEqual(model, original);
  assert.deepEqual(bytes, before);
  model.nodes[1].translation[0] = 999;
  bytes.fill(0);
  const p = createAnimationPlayer(d);
  p.sample(1);
  close(p.translations.subarray(3, 6), [11, 1, 0]);
});

const failures = [
  (f) => {
    f.model.asset.version = "1.0";
  },
  (f) => {
    f.model.bufferViews[0].byteLength = 4;
  },
  (f) => {
    f.model.bufferViews[0].byteOffset = 1;
  },
  (f) => {
    f.model.bufferViews[1].byteStride = 8;
  },
  (f) => {
    f.model.bufferViews[1].byteStride = 14;
  },
  (f) => {
    f.model.bufferViews[0].byteLength = f.bytes.length + 1;
  },
  (f) => {
    f.model.buffers[0].byteLength = f.bytes.length + 1;
  },
  (f) => {
    f.model.accessors[0].count = Number.MAX_SAFE_INTEGER;
  },
  (f) => {
    f.model.accessors[0].componentType = 5125;
  },
  (f) => {
    delete f.model.accessors[0].min;
  },
  (f) => {
    f.model.accessors[1].type = "VEC4";
  },
  (f) => {
    f.model.accessors[1].normalized = true;
  },
  (f) => {
    new DataView(f.bytes.buffer).setFloat32(0, Infinity, true);
  },
  (f) => {
    f.model.accessors[0].extensions = { EXT_unknown: {} };
  },
  (f) => {
    f.model.bufferViews[0].extensions = { EXT_meshopt_compression: {} };
  },
  (f) => {
    f.model.animations[0].channels[0].target.extensions = {
      KHR_animation_pointer: { pointer: "/nodes/1/translation" },
    };
  },
  (f) => {
    f.model.animations[0].channels[0].target.path = "unknown";
  },
  (f) => {
    f.model.nodes[0].children = [2];
  },
  (f) => {
    f.model.nodes[1].children = [2, 2];
  },
  (f) => {
    f.model.nodes[1].children = [1];
  },
  (f) => {
    f.model.nodes[0].weights = [1];
  },
  (f) => {
    f.model.nodes[1].weights = [1];
  },
  (f) => {
    f.model.meshes[0].primitives.push({ targets: [{}] });
  },
  (f) => {
    f.model.skins[0].inverseBindMatrices = 0;
  },
  (f) => {
    f.model.skins[0].joints = [99];
  },
  (f) => {
    delete f.model.nodes[0].mesh;
  },
  (f) => {
    f.model.nodes[0].extensions = { EXT_mesh_gpu_instancing: {} };
  },
  (f) => {
    const v = f.model.bufferViews[f.indices.sparseIndices];
    f.bytes[v.byteOffset] = 2;
    f.bytes[v.byteOffset + 1] = 1;
  },
  (f) => {
    const v = f.model.bufferViews[f.indices.sparseIndices];
    f.bytes[v.byteOffset + 1] = 4;
  },
  (f) => {
    f.model.bufferViews[f.indices.sparseIndices].byteStride = 4;
  },
  (f) => {
    f.model.accessors[f.indices.weights].sparse.count = 5;
  },
  (f) => {
    f.model.accessors[f.indices.weights].byteOffset = 4;
  },
];
for (let i = 0; i < failures.length; i++)
  test(`invalid glTF accessor/graph ${i} is rejected`, () => {
    const f = animationFixture();
    failures[i](f);
    assert.throws(() => decodeGltfAnimation(f.model, [f.bytes]), AnimationPoseError);
  });

test("component budget and missing/shared input buffers fail before player construction", () => {
  const f = animationFixture();
  assert.throws(() => decodeGltfAnimation(f.model, [], {}), { code: "GLTF_ANIMATION_BUFFER" });
  assert.throws(() => decodeGltfAnimation(f.model, [f.bytes], { maxComponents: 2 }), {
    code: "GLTF_ANIMATION_LIMIT",
  });
  assert.throws(
    () => decodeGltfAnimation(f.model, [new Uint8Array(new SharedArrayBuffer(f.bytes.length))]),
    { code: "GLTF_ANIMATION_BUFFER" },
  );
});
