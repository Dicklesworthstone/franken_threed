import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createGpuAnimationRenderer } from "./animation_render.mjs";

const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const UV = [0, 0, 1, 0, 0, 1];
const fields = [
  "baseColorTexture",
  "metallicRoughnessTexture",
  "normalTexture",
  "emissiveTexture",
  "occlusionTexture",
];
const texture = () => ({ view: {}, sampler: {} });
// Production renderer and lighting/shadow receivers, recording only the GPU
// boundary. Shader-source/packing assertions are NOT native shader execution.
function device(limits = {}) {
  const buffers = [],
    pipelines = [],
    groups = [],
    passes = [],
    writes = [];
  let submissions = 0,
    depth = 0,
    lose;
  const d = {
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 2 ** 26,
      maxUniformBufferBindingSize: 65536,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8,
      maxVertexAttributes: 16,
      maxVertexBufferArrayStride: 2048,
      maxInterStageShaderVariables: 16,
      maxSamplersPerShaderStage: 16,
      maxSampledTexturesPerShaderStage: 16,
      maxTextureDimension2D: 4096,
      ...limits,
    },
    lost: new Promise((r) => {
      lose = r;
    }),
    pushErrorScope() {
      depth++;
    },
    popErrorScope() {
      assert.ok(depth > 0);
      depth--;
      return Promise.resolve(null);
    },
    createBuffer(desc) {
      const data = new ArrayBuffer(desc.size);
      const b = {
        ...desc,
        data,
        destroyed: 0,
        getMappedRange: () => data,
        unmap() {},
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    createBindGroupLayout: (x) => x,
    createPipelineLayout: (x) => x,
    createShaderModule: (x) => x,
    createBindGroup(x) {
      groups.push(x);
      return x;
    },
    async createRenderPipelineAsync(x) {
      pipelines.push(x);
      return x;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          const p = { desc, draws: [] };
          passes.push(p);
          let pipeline;
          const bindings = new Map(),
            vertices = new Map();
          return {
            setPipeline(x) {
              pipeline = x;
            },
            setBindGroup(i, g, offsets = []) {
              bindings.set(i, { group: g, offsets });
            },
            setVertexBuffer(i, b) {
              vertices.set(i, b);
            },
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw(...args) {
              p.draws.push({
                pipeline,
                bindings: new Map(bindings),
                vertices: new Map(vertices),
                args,
              });
            },
            drawIndexed(...args) {
              this.draw(...args);
            },
            end() {},
          };
        },
        finish: () => ({}),
      };
    },
    queue: {
      writeBuffer(buffer, offset, value, start = 0, length) {
        const width = value.BYTES_PER_ELEMENT ?? 1,
          storage = ArrayBuffer.isView(value) ? value.buffer : value;
        const bytes = new Uint8Array(
          storage,
          (value.byteOffset ?? 0) + start * width,
          (length ?? value.byteLength / width - start) * width,
        ).slice();
        new Uint8Array(buffer.data, offset, bytes.length).set(bytes);
        writes.push({ buffer, bytes });
      },
      submit() {
        submissions++;
      },
      onSubmittedWorkDone: async () => {},
    },
  };
  return {
    d,
    buffers,
    pipelines,
    groups,
    passes,
    writes,
    lose,
    counts: () => [buffers.length, pipelines.length, groups.length, writes.length, submissions],
    get depth() {
      return depth;
    },
  };
}
const gpu = (tangents = false) => ({
  vertexBuffer: {},
  vertexCount: 3,
  worldMatrix: I(),
  disposed: false,
  failed: false,
  whenIdle: async () => {},
  vertexLayout: {
    arrayStride: 40,
    stepMode: "vertex",
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      ...(tangents ? [{ shaderLocation: 2, offset: 24, format: "float32x4" }] : []),
    ],
  },
});
function environment(d) {
  const snapshot = Object.freeze({
    profile: "f3d-animation-environment-v1",
    version: 1,
    diffuseView: {},
    specularView: {},
    brdfView: {},
    sampler: {},
    mipLevelCount: 3,
  });
  return {
    sample(other) {
      assert.equal(other, d);
      return snapshot;
    },
    whenIdle: async () => {},
  };
}
function shadow(d) {
  const snapshot = Object.freeze({
    view: {},
    sampler: {},
    version: 1,
    width: 8,
    height: 8,
    viewProjection: I(),
  });
  return {
    sample(other) {
      assert.equal(other, d);
      return snapshot;
    },
    whenIdle: async () => {},
  };
}
const frame = (draws, extra = {}) => ({
  colorView: {},
  depthView: {},
  viewProjection: I(),
  draws,
  lighting: { cameraPosition: [0, 0, 3], lights: [{ type: "directional" }] },
  ...extra,
});
const shader = (g) => g.passes.at(-1).draws[0].pipeline.fragment.module.code;
const words = (g) =>
  new Float32Array(
    g.writes.findLast((w) => w.buffer.size === 256 || w.buffer.label === "arena").bytes.buffer,
  );
const code = (name) => ({ code: "ANIMATION_RENDER_" + name });

for (const shading of ["lambert", "metallic-roughness"])
  for (const strength of [0, 0.25, 1])
    test(`${shading} occlusion strength ${strength} packs without enlarging the uniform`, async () => {
      const g = device(),
        r = await createGpuAnimationRenderer(g.d, {
          environment: true,
          maxDraws: 1,
          maxBytes: 936,
        });
      const map = texture(),
        m = await r.addMesh(gpu(), {
          shading,
          texCoords: UV,
          occlusionTexture: map,
          occlusionStrength: strength,
        });
      assert.equal(r.allocatedBytes, 936);
      r.render(frame([m], { environment: { map: environment(g.d) } }));
      assert.equal(words(g)[51], strength);
      assert.equal(words(g).length, 64);
      assert.deepEqual(g.passes.at(-1).draws[0].bindings.get(1).group.entries, [
        { binding: 8, resource: map.sampler },
        { binding: 9, resource: map.view },
      ]);
      assert.match(
        shader(g),
        /struct OcclusionNormal \{ x: vec3<f32>, strength: f32, y: vec3<f32>, pad0: f32, z: vec3<f32>, pad1: f32 \}/,
      );
      assert.match(shader(g), /normal_from_local: OcclusionNormal/);
      assert.equal(g.depth, 0);
      await r.whenIdle();
      r.dispose();
      assert.ok(g.buffers.every((b) => b.destroyed === 1));
    });

test("the emitted scalar follows glTF R/strength; only indirect light is multiplied", async () => {
  const g = device(),
    r = await createGpuAnimationRenderer(g.d, { environment: true, maxDraws: 1 });
  const m = await r.addMesh(gpu(), {
    shading: "metallic-roughness",
    texCoords: UV,
    occlusionTexture: texture(),
    emissiveFactor: [1, 2, 3],
  });
  r.render(frame([m], { environment: { map: environment(g.d) } }));
  const s = shader(g),
    expression = /let occlusion = ([^;]+);/.exec(s)[1];
  // Evaluate only the trusted emitted arithmetic scalar, not a copied formula or
  // a WGSL interpreter. Vector lighting and native pixels are tested elsewhere.
  const scalar = new Function("draw_info", "occlusion_texel", `return (${expression});`);
  for (const strength of [0, 0.25, 1])
    for (const red of [0, 0.25, 1]) {
      const actual = scalar({ normal_from_local: { strength } }, { r: red, g: 999, b: -777, a: 0 });
      assert.equal(actual, 1 - strength + strength * red);
    }
  const body = s.slice(s.indexOf("fn illuminate"), s.indexOf("struct VertexOutput"));
  assert.match(body, /var result = emission;/);
  assert.match(body, /result \+= environment_lighting\([^;]+\) \* occlusion;/);
  const direct = body.slice(body.indexOf("for (var i = 0u;"));
  assert.doesNotMatch(direct, /occlusion/);
  assert.match(s, /let rgba = draw_info.color \* input.color/);
  assert.doesNotMatch(s, /occlusion_texel\.[gba]/);
  assert.equal(words(g)[51], 1);
  r.dispose();
});

for (const tangent of [false, true])
  test(`all five maps retain independent UVs with shadows, IBL and ${tangent ? "authored" : "derivative"} normals`, async () => {
    const g = device(),
      r = await createGpuAnimationRenderer(g.d, { environment: true, shadows: true, maxDraws: 1 });
    const maps = Object.fromEntries(fields.map((f) => [f, texture()]));
    const coordinates = Object.fromEntries(
      fields.map((f, i) => [f, { texCoords: UV, uvTransform: [i + 1, 0, 0, i + 2, 0.25, 0.5] }]),
    );
    const m = await r.addMesh(gpu(tangent), {
      shading: "metallic-roughness",
      ...maps,
      mapCoordinates: coordinates,
      occlusionStrength: 0.5,
      normalScale: 0.75,
      alphaMode: "MASK",
      baseColor: [1, 1, 1, 0.6],
    });
    const world = I();
    world[0] = -2;
    world[5] = 4;
    world[10] = 8;
    r.render(
      frame([{ mesh: m, worldMatrix: world }], {
        environment: { map: environment(g.d) },
        shadow: { map: shadow(g.d) },
      }),
    );
    const draw = g.passes.at(-1).draws[0],
      s = shader(g),
      u = words(g);
    assert.equal(draw.pipeline.vertex.buffers[1].arrayStride, 64);
    assert.deepEqual(
      draw.bindings.get(1).group.entries.map((e) => e.binding),
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    );
    assert.equal(draw.bindings.get(2).group.entries.length, 9);
    assert.match(s, /occlusion_texture, occlusion_sampler, input\.uv_4/);
    assert.match(s, /@location\(9\) uv_4/);
    assert.ok(s.indexOf("let occlusion_texel") < s.indexOf("discard;"));
    assert.equal(u[27], 0.75);
    assert.equal(u[31], -1);
    assert.equal(u[51], 0.5);
    assert.equal(u[48], -0.5);
    assert.equal(u[53], 0.25);
    assert.equal(u[58], 0.125);
    const surface = new Float32Array(draw.vertices.get(1).data);
    assert.deepEqual([...surface.slice(14, 16)], [0.25, 0.5]);
    assert.deepEqual([...surface.slice(30, 32)], [5.25, 0.5]);
    assert.equal(s.includes("dpdx(input.world)"), !tangent);
    await r.whenIdle();
    r.dispose();
  });

test("draw strengths are separate snapshots and invalid final draw publishes nothing", async () => {
  const g = device({ minUniformBufferOffsetAlignment: 512 }),
    r = await createGpuAnimationRenderer(g.d, { environment: true, maxDraws: 2, label: "arena" });
  const options = {
    shading: "lambert",
    texCoords: UV,
    occlusionTexture: texture(),
    occlusionStrength: 0.25,
  };
  const pending = r.addMesh(gpu(), options);
  options.occlusionStrength = 0.75;
  const m = await pending;
  const env = environment(g.d);
  r.render(frame([{ mesh: m, occlusionStrength: 0 }, m], { environment: { map: env } }));
  const arena = g.writes.findLast((w) => w.buffer.label === "arena"),
    u = new Float32Array(arena.bytes.buffer);
  assert.equal(u[51], 0);
  assert.equal(u[128 + 51], 0.25);
  const before = g.counts();
  assert.throws(
    () => r.render(frame([m, { mesh: m, occlusionStrength: 1.01 }], { environment: { map: env } })),
    code("VALUE"),
  );
  assert.deepEqual(g.counts(), before);
  assert.equal(r.version, 1);
  r.render(frame([m], { environment: { map: env } }));
  assert.equal(
    new Float32Array(g.writes.findLast((w) => w.buffer.label === "arena").bytes.buffer)[51],
    0.25,
  );
  r.dispose();
});

for (const strength of [-1, 1.01, NaN, Infinity, "1", {}, false])
  test(`invalid material/draw strength ${String(strength)} fails before GPU writes`, async () => {
    const g = device(),
      r = await createGpuAnimationRenderer(g.d, { maxDraws: 1 }),
      before = g.counts();
    await assert.rejects(
      r.addMesh(gpu(), {
        shading: "lambert",
        texCoords: UV,
        occlusionTexture: texture(),
        occlusionStrength: strength,
      }),
      code("VALUE"),
    );
    assert.deepEqual(g.counts(), before);
    const m = await r.addMesh(gpu(), {
        shading: "lambert",
        texCoords: UV,
        occlusionTexture: texture(),
      }),
      ready = g.counts();
    assert.throws(() => r.render(frame([{ mesh: m, occlusionStrength: strength }])), code("VALUE"));
    assert.deepEqual(g.counts(), ready);
    r.dispose();
  });

test("unlit/depth routes reject occlusion; strengths require a map", async () => {
  for (const format of ["rgba8unorm", null]) {
    const g = device(),
      r = await createGpuAnimationRenderer(g.d, { format, maxDraws: 1 }),
      before = g.counts();
    await assert.rejects(
      r.addMesh(gpu(), { texCoords: UV, occlusionTexture: texture() }),
      code("OPTIONS"),
    );
    await assert.rejects(r.addMesh(gpu(), { occlusionStrength: 0 }), code("OPTIONS"));
    assert.deepEqual(g.counts(), before);
    r.dispose();
  }
  const g = device(),
    r = await createGpuAnimationRenderer(g.d, { maxDraws: 1 }),
    m = await r.addMesh(gpu(), { shading: "lambert" }),
    before = g.counts();
  assert.throws(() => r.render(frame([{ mesh: m, occlusionStrength: 0 }])), code("OPTIONS"));
  assert.deepEqual(g.counts(), before);
  r.dispose();
});

test("direct-only frames do not apply occlusion even when its material map is present", async () => {
  const g = device(),
    r = await createGpuAnimationRenderer(g.d, { environment: true, maxDraws: 1 }),
    env = environment(g.d);
  const m = await r.addMesh(gpu(), {
    shading: "lambert",
    texCoords: UV,
    occlusionTexture: texture(),
  });
  r.render(frame([m], { environment: { map: env } }));
  assert.match(shader(g), /\* occlusion;/);
  const before = g.counts();
  r.render(frame([m]));
  assert.doesNotMatch(shader(g), /let occlusion =|\* occlusion;|fn environment_lighting/);
  assert.equal(g.counts()[0], before[0]);
  assert.equal(g.counts()[1], before[1]);
  r.dispose();
});

for (const limits of [
  { maxSamplersPerShaderStage: 6 },
  { maxSampledTexturesPerShaderStage: 8 },
  { maxVertexAttributes: 9 },
  { maxInterStageShaderVariables: 9 },
  { maxVertexBufferArrayStride: 63 },
])
  test(
    "five-map aggregate limits reject before allocations " + JSON.stringify(limits),
    async () => {
      const g = device(limits),
        r = await createGpuAnimationRenderer(g.d, {
          environment: true,
          shadows: true,
          maxDraws: 1,
        }),
        before = g.counts();
      const maps = Object.fromEntries(fields.map((f) => [f, texture()]));
      await assert.rejects(
        r.addMesh(gpu(true), {
          shading: "metallic-roughness",
          ...maps,
          mapCoordinates: Object.fromEntries(fields.map((f) => [f, { texCoords: UV }])),
        }),
        code("LIMIT"),
      );
      assert.deepEqual(g.counts(), before);
      r.dispose();
    },
  );

test("one byte below the occlusion surface budget fails; malformed texture/UV descriptors have no effects", async () => {
  const g = device(),
    r = await createGpuAnimationRenderer(g.d, { environment: true, maxDraws: 1, maxBytes: 935 }),
    before = g.counts();
  await assert.rejects(
    r.addMesh(gpu(), { shading: "lambert", texCoords: UV, occlusionTexture: texture() }),
    code("LIMIT"),
  );
  assert.deepEqual(g.counts(), before);
  r.dispose();
  for (const material of [
    { occlusionTexture: {} },
    { occlusionTexture: { view: {}, sampler: {}, strength: 1 } },
    { occlusionTexture: texture(), texCoords: null },
    { occlusionTexture: texture(), mapCoordinates: { occlusionTexture: { texCoords: [0, 0] } } },
  ]) {
    const g = device(),
      r = await createGpuAnimationRenderer(g.d, { maxDraws: 1 }),
      before = g.counts();
    await assert.rejects(r.addMesh(gpu(), { shading: "lambert", texCoords: UV, ...material }));
    assert.deepEqual(g.counts(), before);
    r.dispose();
  }
});

test("shared ORM resources stay borrowed and device loss releases only renderer buffers", async () => {
  const g = device(),
    r = await createGpuAnimationRenderer(g.d, { environment: true, maxDraws: 1 }),
    orm = texture();
  const m = await r.addMesh(gpu(), {
    shading: "metallic-roughness",
    texCoords: UV,
    metallicRoughnessTexture: orm,
    occlusionTexture: orm,
  });
  r.render(frame([m], { environment: { map: environment(g.d) } }));
  const entries = g.passes.at(-1).draws[0].bindings.get(1).group.entries;
  assert.equal(entries[1].resource, entries[3].resource);
  assert.equal(entries[0].resource, entries[2].resource);
  g.lose({ message: "lost" });
  await assert.rejects(r.whenIdle(), code("LOST"));
  assert.equal(r.allocatedBytes, 0);
  r.dispose();
  assert.ok(g.buffers.every((b) => b.destroyed === 1));
  assert.deepEqual(orm, texture());
});

// Golden generated by the unchanged renderer blob ade9d51b25c2284246f20a1fbbc47c48f47b1d35.
// All 16 pre-existing map masks, both normal paths, shared/independent UVs,
// and direct/shadow/IBL combinations retain identical WGSL and draw bytes.

async function defaultDigest() {
  const digest = createHash("sha256");
  for (const tangents of [false, true])
    for (let mask = 0; mask < 16; mask++) {
      const g = device(),
        r = await createGpuAnimationRenderer(g.d, {
          environment: true,
          shadows: true,
          maxDraws: 1,
        });
      const maps = Object.fromEntries(
        fields
          .slice(0, 4)
          .filter((_, i) => mask & (1 << i))
          .map((f) => [f, texture()]),
      );
      for (const independent of [false, true]) {
        const m = await r.addMesh(gpu(tangents), {
          shading: "metallic-roughness",
          texCoords: UV,
          ...maps,
          ...(independent
            ? {
                mapCoordinates: Object.fromEntries(
                  Object.keys(maps).map((f) => [
                    f,
                    { texCoords: UV, uvTransform: [2, 0, 0, 3, 0.25, 0.5] },
                  ]),
                ),
              }
            : {}),
        });
        for (const ambient of [false, true])
          for (const projected of [false, true]) {
            r.render(
              frame([m], {
                ...(ambient ? { environment: { map: environment(g.d) } } : {}),
                ...(projected ? { shadow: { map: shadow(g.d) } } : {}),
              }),
            );
            digest.update(shader(g));
            digest.update(words(g));
          }
        m.dispose();
      }
      r.dispose();
    }
  return digest.digest("hex");
}

test("256 existing non-occlusion shader/uniform combinations remain byte-identical", async () => {
  assert.equal(
    await defaultDigest(),
    "ae0e6dd923ae3563b2d5fb1cbe877043a5c6e70c66bfdb2225bc08499f8edd72",
  );
});
