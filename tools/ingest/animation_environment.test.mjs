import assert from "node:assert/strict";
import test from "node:test";
import {
  animationEnvironmentShader,
  createGpuAnimationEnvironment,
  planAnimationEnvironment,
} from "./animation_environment.mjs";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((r, j) => {
    resolve = r;
    reject = j;
  });
  return { promise, resolve, reject };
};
const tick = () => new Promise((r) => setImmediate(r));
export function environmentGpu({ compilation, completion } = {}) {
  const textures = [],
    buffers = [],
    events = [],
    scopes = [],
    lost = deferred();
  let depth = 0,
    failTexture = 0;
  function texture(format = "rgba16float", width = 16, height = 8, layers = 1, usage = 4) {
    return {
      format,
      width,
      height,
      depthOrArrayLayers: layers,
      usage,
      dimension: "2d",
      sampleCount: 1,
      destroyed: 0,
      views: [],
      createView(d = {}) {
        const view = { texture: this, descriptor: d };
        this.views.push(view);
        return view;
      },
      destroy() {
        this.destroyed++;
      },
    };
  }
  const device = {
    limits: {
      maxTextureDimension2D: 4096,
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 2 ** 26,
    },
    lost: lost.promise,
    queue: {
      writeBuffer(b, offset, bytes) {
        events.push(["write", new Uint8Array(bytes).slice()]);
      },
      submit(c) {
        events.push(["submit", c]);
      },
      onSubmittedWorkDone: () => completion?.promise ?? Promise.resolve(),
    },
    pushErrorScope(k) {
      depth++;
      events.push(["push", k]);
    },
    popErrorScope() {
      depth--;
      return scopes.shift() ?? Promise.resolve(null);
    },
    createTexture(d) {
      if (failTexture === textures.length + 1) throw new Error("texture allocation");
      const t = texture(d.format, d.size[0], d.size[1], d.size[2], d.usage);
      t.mipLevelCount = d.mipLevelCount;
      textures.push(t);
      return t;
    },
    createBuffer(d) {
      const b = {
        ...d,
        destroyed: 0,
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    createSampler(d) {
      return d;
    },
    createBindGroupLayout: (d) => d,
    createPipelineLayout: (d) => d,
    createShaderModule(d) {
      events.push(["shader", d]);
      return d;
    },
    createRenderPipelineAsync: (d) => compilation?.promise ?? Promise.resolve(d),
    createBindGroup(d) {
      events.push(["group", d]);
      return d;
    },
    createCommandEncoder: () => ({
      beginRenderPass(d) {
        events.push(["pass", d]);
        return {
          setPipeline() {},
          setBindGroup(...x) {
            events.push(["bind", ...x]);
          },
          draw(n) {
            events.push(["draw", n]);
          },
          end() {
            events.push(["end"]);
          },
        };
      },
      finish: () => ({}),
    }),
  };
  return {
    device,
    textures,
    buffers,
    events,
    scopes,
    lost,
    texture,
    get depth() {
      return depth;
    },
    set failTexture(v) {
      failTexture = v;
    },
  };
}
const small = { size: 8, diffuseSize: 2, lutSize: 4, samples: 32 };
test("plan has one immutable record per face/mip and a separate DFG pass", () => {
  const p = planAnimationEnvironment(small);
  assert.equal(p.levels, 4);
  assert.equal(p.passes.length, 31);
  assert.deepEqual(
    p.passes.slice(0, 6).map((x) => x.face),
    [0, 1, 2, 3, 4, 5],
  );
  assert.deepEqual(
    p.passes.filter((x) => x.face === 0 && x.kind === 0).map((x) => x.roughness),
    [0, 1 / 3, 2 / 3, 1],
  );
  assert.equal(p.textureBytes, (6 * (64 + 16 + 4 + 1) + 6 * 4 + 16) * 8);
  assert.equal(p.sampleWork, 6 * 64 + (6 * (16 + 4 + 1) + 6 * 4 + 16) * 32);
  assert.ok(Object.isFrozen(p) && Object.isFrozen(p.passes) && p.passes.every(Object.isFrozen));
  assert.equal(p.uniformBytes, 31 * 256);
  assert.equal(planAnimationEnvironment({ ...small, alignment: 16 }).stride, 32);
});
for (const options of [
  { size: 7 },
  { samples: 0 },
  { samples: 4097 },
  { maxTextureBytes: 1 },
  { maxSampleWork: 1 },
  { alignment: 48 },
  { lutSize: 1 },
  { diffuseSize: 0 },
])
  test("work/size limits reject rather than reducing quality " + JSON.stringify(options), () =>
    assert.throws(() => planAnimationEnvironment({ ...small, ...options }), {
      code: "ANIMATION_ENVIRONMENT_LIMIT",
    }),
  );
for (const cube of [false, true])
  test(
    (cube ? "cubemap" : "panorama") +
      " produces all irradiance/specular/LUT targets with distinct uniform ranges",
    async () => {
      const g = environmentGpu(),
        source = g.texture("rgba16float", 16, cube ? 16 : 8, cube ? 6 : 1),
        env = await createGpuAnimationEnvironment(g.device, source, small);
      assert.equal(env.mipLevelCount, 4);
      assert.equal(env.textureBytes, env.plan.textureBytes);
      assert.equal(g.textures.length, 3);
      assert.deepEqual(
        g.textures.map((t) => [t.width, t.depthOrArrayLayers, t.mipLevelCount, t.usage]),
        [
          [8, 6, 4, 20],
          [2, 6, 1, 20],
          [4, 1, 1, 20],
        ],
      );
      assert.equal(env.diffuseView.descriptor.dimension, "cube");
      assert.equal(env.specularView.descriptor.dimension, "cube");
      assert.equal(g.events.filter((e) => e[0] === "submit").length, 1);
      assert.equal(g.events.filter((e) => e[0] === "pass").length, 31);
      const writes = g.events.filter((e) => e[0] === "write");
      assert.equal(writes.length, 1);
      const bytes = new DataView(writes[0][1].buffer);
      for (const [i, p] of env.plan.passes.entries())
        assert.deepEqual(
          [
            bytes.getUint32(i * 256, true),
            bytes.getUint32(i * 256 + 4, true),
            bytes.getUint32(i * 256 + 8, true),
            bytes.getUint32(i * 256 + 12, true),
          ],
          [p.face, p.kind, p.samples, p.width],
        );
      assert.deepEqual(
        g.events.filter((e) => e[0] === "bind").map((e) => e[3][0]),
        Array.from({ length: 31 }, (_, i) => i * 256),
      );
      assert.equal(g.depth, 0);
      assert.equal(g.buffers[0].destroyed, 1);
      assert.equal(source.destroyed, 0);
      const input = g.events.find((e) => e[0] === "group")[1].entries;
      assert.equal(input[1].resource.addressModeU, cube ? "clamp-to-edge" : "repeat");
      assert.equal(input[2].resource.descriptor.dimension, cube ? "cube" : "2d");
      env.dispose();
      env.dispose();
      assert.equal(env.textureBytes, 0);
      assert.ok(g.textures.every((t) => t.destroyed === 1));
      assert.equal(source.destroyed, 0);
    },
  );
test("invalid source and options fail before any GPU effects", async () => {
  const g = environmentGpu();
  for (const source of [
    null,
    { ...g.texture(), sampleCount: 4 },
    g.texture("rgba8unorm-srgb"),
    g.texture("rgba16float", 16, 7),
    g.texture("rgba16float", 16, 8, 6),
    g.texture("rgba16float", 16, 8, 1, 16),
  ])
    await assert.rejects(createGpuAnimationEnvironment(g.device, source, small));
  await assert.rejects(
    createGpuAnimationEnvironment(g.device, g.texture(), { ...small, guessFlip: true }),
  );
  assert.equal(g.events.length, 0);
});
test("compilation scopes are closed while waiting and no texture exists until compilation succeeds", async () => {
  const compilation = deferred(),
    g = environmentGpu({ compilation }),
    pending = createGpuAnimationEnvironment(g.device, g.texture(), small);
  assert.equal(g.depth, 0);
  assert.equal(g.textures.length, 0);
  compilation.resolve({});
  const env = await pending;
  env.dispose();
});
test("compilation failure creates no owned GPU allocations", async () => {
  const compilation = deferred(),
    g = environmentGpu({ compilation }),
    pending = createGpuAnimationEnvironment(g.device, g.texture(), small),
    error = new Error("WGSL");
  compilation.reject(error);
  await assert.rejects(pending, (e) => e === error);
  assert.equal(g.textures.length + g.buffers.length, 0);
});
test("completion is awaited before returning views or retiring filter uniforms", async () => {
  const completion = deferred(),
    g = environmentGpu({ completion });
  let published = false;
  const pending = createGpuAnimationEnvironment(g.device, g.texture(), small).then((x) => {
    published = true;
    return x;
  });
  await tick();
  assert.equal(published, false);
  assert.equal(g.buffers[0].destroyed, 0);
  completion.resolve();
  const env = await pending;
  assert.equal(published, true);
  assert.equal(g.buffers[0].destroyed, 1);
  env.dispose();
});
test("partial allocation failure destroys only owned textures", async () => {
  const g = environmentGpu(),
    source = g.texture();
  g.failTexture = 2;
  await assert.rejects(
    createGpuAnimationEnvironment(g.device, source, small),
    /texture allocation/,
  );
  assert.equal(g.textures.length, 1);
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(source.destroyed, 0);
});
test("driver validation errors reject without publishing invalid environment views", async () => {
  const g = environmentGpu();
  g.scopes.push(
    Promise.resolve(null),
    Promise.resolve(null),
    Promise.resolve({ message: "invalid cube" }),
    Promise.resolve(null),
  );
  await assert.rejects(createGpuAnimationEnvironment(g.device, g.texture(), small), {
    code: "ANIMATION_ENVIRONMENT_GPU",
  });
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  assert.equal(g.buffers[0].destroyed, 1);
  assert.equal(g.depth, 0);
});
test("abort promptly rejects unresolved compilation and ignores its late result", async () => {
  const compilation = deferred(),
    g = environmentGpu({ compilation }),
    c = new AbortController(),
    pending = createGpuAnimationEnvironment(g.device, g.texture(), { ...small, signal: c.signal });
  c.abort();
  await assert.rejects(pending, { name: "AbortError" });
  compilation.resolve({});
  await tick();
  assert.equal(g.textures.length, 0);
});
test("abort during queued work releases owned allocations without claiming GPU rollback", async () => {
  const completion = deferred(),
    g = environmentGpu({ completion }),
    c = new AbortController(),
    source = g.texture();
  const pending = createGpuAnimationEnvironment(g.device, source, { ...small, signal: c.signal });
  await tick();
  c.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  assert.equal(g.buffers[0].destroyed, 1);
  assert.equal(source.destroyed, 0);
  completion.resolve();
});
test("device loss before completion rejects; after publication invalidates and releases the environment", async () => {
  const completion = deferred(),
    g = environmentGpu({ completion }),
    pending = createGpuAnimationEnvironment(g.device, g.texture(), small);
  await tick();
  g.lost.resolve({ message: "lost" });
  await assert.rejects(pending, { code: "ANIMATION_ENVIRONMENT_LOST" });
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  completion.resolve();
  const h = environmentGpu(),
    env = await createGpuAnimationEnvironment(h.device, h.texture(), small);
  h.lost.resolve({ message: "lost later" });
  await tick();
  assert.equal(env.failed, true);
  assert.equal(env.textureBytes, 0);
  env.dispose();
});
test("successful preparation releases its construction AbortSignal without owning caller lifetime", async () => {
  const c = new AbortController(),
    g = environmentGpu(),
    env = await createGpuAnimationEnvironment(g.device, g.texture(), {
      ...small,
      signal: c.signal,
    });
  c.abort();
  assert.equal(env.failed, false);
  assert.ok(env.textureBytes > 0);
  env.dispose();
});
test("shader profiles use explicit LOD, stable cube faces, correlated visibility and split-sum weighting", () => {
  for (const cube of [false, true]) {
    const shader = animationEnvironmentShader(cube);
    assert.doesNotMatch(shader, /SOURCE_DECLARATION|SOURCE_SAMPLE/);
    assert.match(shader, cube ? /texture_cube<f32>/ : /texture_2d<f32>/);
    assert.match(shader, /textureSampleLevel/);
    assert.doesNotMatch(shader, /textureSample\(/);
    assert.match(shader, /4\.0\*visibility\*nl\*vh\/h\.z/);
    assert.match(shader, /info\.kind==2u/);
  }
  assert.match(animationEnvironmentShader(false), /atan2\(d.z,d.x\)/);
});
