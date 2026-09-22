import assert from "node:assert/strict";
import test from "node:test";
import { createGpuAnimationEnvironment } from "./animation_environment.mjs";
import {
  ENVIRONMENT_UNIFORM_BYTES,
  environmentLightingWgsl,
  packAnimationEnvironment,
} from "./animation_environment_receiver.mjs";

const I = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
const small = { size: 4, diffuseSize: 2, lutSize: 4, samples: 16 };
const reject = (code, message) => {
  const error = new Error(message);
  error.code = code;
  throw error;
};
const error = { code: "ANIMATION_RENDER_ENVIRONMENT" };
function device() {
  let lose;
  const owned = [],
    events = [];
  const d = {
    owned,
    events,
    lost: new Promise((r) => (lose = r)),
    lose: (info) => lose(info),
    limits: {
      maxTextureDimension2D: 4096,
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 2 ** 26,
    },
    pushErrorScope() {},
    popErrorScope: async () => null,
    createTexture(desc) {
      const t = {
        desc,
        destroyed: false,
        createView: (descriptor) => ({ texture: t, descriptor }),
        destroy() {
          this.destroyed = true;
        },
      };
      owned.push(t);
      return t;
    },
    createBuffer(desc) {
      const b = {
        desc,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      owned.push(b);
      return b;
    },
    createSampler: (x) => x,
    createBindGroupLayout: (x) => x,
    createPipelineLayout: (x) => x,
    createShaderModule: (x) => x,
    createRenderPipelineAsync: async (x) => x,
    createBindGroup: (x) => x,
    createCommandEncoder: () => ({
      beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
      finish: () => ({}),
    }),
    queue: {
      writeBuffer() {
        events.push("write");
      },
      submit() {
        events.push("submit");
      },
      onSubmittedWorkDone: async () => {},
    },
  };
  return d;
}
const source = () => ({
  dimension: "2d",
  width: 8,
  height: 4,
  depthOrArrayLayers: 1,
  format: "rgba16float",
  usage: 4,
  sampleCount: 1,
  destroyed: false,
  createView: () => ({}),
  destroy() {
    this.destroyed = true;
  },
});
async function prepared() {
  const d = device(),
    src = source(),
    map = await createGpuAnimationEnvironment(d, src, small);
  return { d, src, map };
}
test("prepared environments lend a stable frozen same-device snapshot without owning the source", async () => {
  const { d, src, map } = await prepared(),
    s = map.sample(d);
  assert.equal(s, map.sample(d));
  assert.ok(Object.isFrozen(s));
  assert.equal(s.profile, "f3d-animation-environment-v1");
  assert.equal(s.version, 1);
  for (const key of ["diffuseView", "specularView", "brdfView", "sampler", "mipLevelCount"])
    assert.equal(s[key], map[key]);
  assert.equal(await map.whenIdle(), map);
  assert.throws(() => map.sample(device()), { code: "ANIMATION_ENVIRONMENT_DEVICE" });
  map.dispose();
  assert.equal(src.destroyed, false);
  assert.equal(map.textureBytes, 0);
  assert.ok(d.owned.every((x) => x.destroyed));
  assert.throws(() => map.sample(d), { code: "ANIMATION_ENVIRONMENT_DISPOSED" });
  await assert.rejects(map.whenIdle(), { code: "ANIMATION_ENVIRONMENT_DISPOSED" });
});
test("construction cancellation does not revoke an already prepared environment", async () => {
  const d = device(),
    controller = new AbortController(),
    map = await createGpuAnimationEnvironment(d, source(), { ...small, signal: controller.signal });
  const s = map.sample(d);
  controller.abort();
  assert.equal(map.sample(d), s);
  assert.equal(await map.whenIdle(), map);
  map.dispose();
});
test("device loss revokes published snapshots and releases owned textures", async () => {
  const { d, map } = await prepared();
  map.sample(d);
  d.lose({ message: "gone" });
  await Promise.resolve();
  assert.equal(map.failed, true);
  assert.equal(map.textureBytes, 0);
  assert.throws(() => map.sample(d), { code: "ANIMATION_ENVIRONMENT_LOST" });
  await assert.rejects(map.whenIdle(), { code: "ANIMATION_ENVIRONMENT_LOST" });
  map.dispose();
});
test("uniform packing matches padded WGSL mat3 and intensity/mip layout", async () => {
  const { d, map } = await prepared(),
    out = new Float32Array(ENVIRONMENT_UNIFORM_BYTES / 4).fill(9);
  const packed = packAnimationEnvironment(d, { map }, out, reject);
  assert.deepEqual([...out], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0]);
  assert.equal(packed.snapshot, map.sample(d));
  assert.equal(packed.map, map);
  packed.check();
  const rotation = new Float64Array([0, 0, -1, 0, 1, 0, 1, 0, 0]);
  packAnimationEnvironment(d, { map, rotation, intensity: 2.5 }, out, reject);
  rotation.fill(0);
  assert.deepEqual([...out], [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2.5, 2, 0, 0]);
  map.dispose();
  assert.throws(() => packed.check(), { code: "ANIMATION_ENVIRONMENT_DISPOSED" });
});
test("invalid descriptors fail atomically without writing receiver uniforms", async () => {
  const { d, map } = await prepared(),
    out = new Float32Array(16).fill(42);
  const invalid = [
    null,
    [],
    {},
    { map, intensity: -1 },
    { map, intensity: Infinity },
    { map, intensity: 1e40 },
    { map, unknown: true },
    { map, rotation: [1] },
    { map, rotation: [NaN, ...I().slice(1)] },
    { map, rotation: [2, 0, 0, 0, 1, 0, 0, 0, 1] },
    { map, rotation: [-1, 0, 0, 0, 1, 0, 0, 0, 1] },
    { map, rotation: [1, 0, 0, 1, 0, 0, 0, 0, 1] },
  ];
  for (const input of invalid) {
    assert.throws(() => packAnimationEnvironment(d, input, out, reject), error);
    assert.deepEqual([...out], new Array(16).fill(42));
  }
  map.dispose();
});
test("shared, resizable and detached rotation storage is rejected", async () => {
  const { d, map } = await prepared(),
    out = new Float32Array(16);
  const detached = new Float32Array(I());
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const rotation of [
    new Float32Array(new SharedArrayBuffer(36)),
    new Float32Array(new ArrayBuffer(36, { maxByteLength: 72 })),
    detached,
  ])
    assert.throws(() => packAnimationEnvironment(d, { map, rotation }, out, reject), error);
  map.dispose();
});
test("invalid or replaced snapshots cannot reach the submission boundary", async () => {
  const { d, map } = await prepared(),
    s = map.sample(d),
    out = new Float32Array(16).fill(42);
  for (const change of [
    { profile: "cubeuv" },
    { version: 2 },
    { diffuseView: null },
    { mipLevelCount: 1 },
    { mipLevelCount: 17 },
  ]) {
    const invalid = Object.freeze({ ...s, ...change }),
      fake = { sample: () => invalid, whenIdle: async () => {} };
    assert.throws(() => packAnimationEnvironment(d, { map: fake }, out, reject), error);
    assert.equal(out[0], 42);
  }
  let current = s;
  const fake = { sample: () => current, whenIdle: async () => {} },
    p = packAnimationEnvironment(d, { map: fake }, out, reject);
  current = Object.freeze({ ...s });
  assert.throws(() => p.check(), error);
  map.dispose();
});
test("complete receiver shader uses matching LUT axes, explicit LOD and disjoint shadow bindings", () => {
  for (const group of [1, 2]) {
    const code = environmentLightingWgsl(group);
    for (const binding of [4, 5, 6, 7, 8])
      assert.ok(code.includes(`@group(${group}) @binding(${binding})`));
    assert.equal((code.match(/texture_cube<f32>/g) || []).length, 2);
    assert.match(code, /vec2<f32>\(perceptual, nv\)/);
    assert.match(code, /perceptual \* environment_info.options.y/);
    assert.doesNotMatch(code, /textureSample\(/);
    assert.match(code, /f0 \* dfg.x \+ vec3<f32>\(dfg.y\)/);
    assert.match(code, /\(1.0 - metallic\) \* base/);
    assert.doesNotMatch(code, /shadow|emission|lighting.lights/);
    assert.match(code, /reflect\(-view, normal\)/);
  }
});
