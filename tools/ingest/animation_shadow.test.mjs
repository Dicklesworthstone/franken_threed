import assert from "node:assert/strict";
import test from "node:test";
import { createGpuAnimationRenderer } from "./animation_render.mjs";
import { createGpuAnimationShadowMap } from "./animation_shadow.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const errorCode = (code) => (error) => error.code === code;
const tick = () => new Promise((resolve) => setImmediate(resolve));
function gpu() {
  return {
    vertexBuffer: {},
    vertexCount: 3,
    worldMatrix: identity(),
    version: 0,
    poseVersion: 0,
    vertexLayout: {
      arrayStride: 40,
      stepMode: "vertex",
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
      ],
    },
    disposed: false,
    failed: false,
    whenIdle: async () => {},
  };
}
function deviceSpy() {
  let lose;
  const d = {
    buffers: [],
    textures: [],
    pipelines: [],
    passes: [],
    writes: [],
    submissions: [],
    groups: [],
    scopes: [],
    lost: new Promise((resolve) => {
      lose = resolve;
    }),
    lose: (value) => lose(value),
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 64 * 1024 * 1024,
      maxUniformBufferBindingSize: 65536,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8,
      maxVertexAttributes: 16,
      maxInterStageShaderVariables: 16,
      maxVertexBufferArrayStride: 2048,
      maxSamplersPerShaderStage: 16,
      maxSampledTexturesPerShaderStage: 16,
      maxTextureDimension2D: 8192,
    },
    pushErrorScope(x) {
      this.scopes.push(x);
    },
    popErrorScope() {
      assert.ok(this.scopes.pop());
      return Promise.resolve(this.scopeError ?? null);
    },
    createBuffer(desc) {
      const data = new ArrayBuffer(desc.size),
        b = {
          ...desc,
          data,
          destroyed: false,
          getMappedRange: () => data,
          unmap() {},
          destroy() {
            this.destroyed = true;
          },
        };
      this.buffers.push(b);
      return b;
    },
    createTexture(desc) {
      const t = {
        ...desc,
        destroyed: false,
        createView: (options) => ({ texture: t, ...options }),
        destroy() {
          this.destroyed = true;
        },
      };
      this.textures.push(t);
      return t;
    },
    createSampler: (desc) => desc,
    createBindGroupLayout: (desc) => desc,
    createPipelineLayout: (desc) => desc,
    createBindGroup(desc) {
      this.groups.push(desc);
      return desc;
    },
    createShaderModule: (desc) => desc,
    createRenderPipelineAsync(desc) {
      this.pipelines.push(desc);
      return Promise.resolve(desc);
    },
    createCommandEncoder() {
      const encoded = [];
      return {
        beginRenderPass: (desc) => {
          const p = { desc, draws: [] },
            groups = new Map(),
            vertices = new Map();
          let pipeline, index;
          d.passes.push(p);
          encoded.push(p);
          const draw = (indexed, args) =>
            p.draws.push({
              indexed,
              args,
              pipeline,
              groups: new Map(groups),
              vertices: new Map(vertices),
              index,
            });
          return {
            setPipeline(x) {
              pipeline = x;
            },
            setBindGroup(slot, group, offsets = []) {
              groups.set(slot, { group, offsets });
            },
            setVertexBuffer(slot, buffer) {
              vertices.set(slot, buffer);
            },
            setIndexBuffer(buffer, format) {
              index = { buffer, format };
            },
            draw: (...args) => draw(false, args),
            drawIndexed: (...args) => draw(true, args),
            setViewport() {},
            setScissorRect() {},
            end() {
              p.ended = true;
            },
          };
        },
        finish: () => encoded,
      };
    },
  };
  d.queue = {
    writeBuffer(buffer, offset, values, start = 0, length = values.length - start) {
      const bytes = values.BYTES_PER_ELEMENT;
      const data = new Uint8Array(
        values.buffer,
        values.byteOffset + start * bytes,
        length * bytes,
      ).slice();
      new Uint8Array(buffer.data, offset, data.length).set(data);
      d.writes.push({ buffer, data });
    },
    submit(commands) {
      for (const passes of commands)
        for (const pass of passes)
          for (const draw of pass.draws) {
            const { group, offsets } = draw.groups.get(0);
            draw.uniform = new Float32Array(
              group.entries[0].resource.buffer.data,
              offsets[0],
              64,
            ).slice();
          }
      d.submissions.push(commands);
    },
    onSubmittedWorkDone: async () => {},
  };
  return d;
}

test("depth-only pipelines and passes have no color attachments or outputs", async () => {
  const d = deviceSpy(),
    r = await createGpuAnimationRenderer(d, { format: null, maxDraws: 2 });
  assert.equal(r.format, null);
  assert.equal(d.pipelines.length, 3);
  for (const p of d.pipelines) {
    assert.deepEqual(p.fragment.targets, []);
    assert.equal(p.depthStencil.depthWriteEnabled, true);
    assert.match(p.fragment.module.code, /@fragment fn fragment_main\(input: VertexOutput\) \{/);
    assert.doesNotMatch(p.fragment.module.code, /return vec4<f32>\(rgb/);
  }
  const mesh = await r.addMesh(gpu());
  r.render({ depthView: {}, viewProjection: identity(), draws: [mesh] });
  assert.deepEqual(d.passes[0].desc.colorAttachments, []);
  assert.equal(d.passes[0].draws.length, 1);
  await r.whenIdle();
  r.dispose();
  assert.ok(d.buffers.every((b) => b.destroyed));
});
test("depth-only MASK keeps texture/vertex alpha, independent UVs and draw ranges", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 16, maxDraws: 2 }),
    g = gpu();
  const map = { view: {}, sampler: {} },
    source = [0, 1, 2];
  const mesh = await s.addMesh(g, {
    indices: source,
    alphaMode: "MASK",
    alphaCutoff: 0.3,
    baseColor: [1, 1, 1, 0.75],
    vertexColors: [1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1],
    baseColorTexture: map,
    mapCoordinates: {
      baseColorTexture: { texCoords: [0, 0, 1, 0, 0, 1], uvTransform: [2, 0, 0, 2, 0.1, 0.2] },
    },
  });
  source.fill(0);
  const reflected = identity();
  reflected[0] = -1;
  s.render({
    viewProjection: identity(),
    draws: [{ mesh, worldMatrix: reflected, first: 1, count: 2 }],
  });
  const draw = d.passes[0].draws[0],
    wgsl = draw.pipeline.fragment.module.code;
  assert.match(wgsl, /color_texel = textureSample\(color_texture, color_sampler, input.uv_0\)/);
  assert.match(wgsl, /draw_info.color \* input.color \* color_texel/);
  assert.match(wgsl, /rgba.a < draw_info.options.x/);
  assert.equal(draw.pipeline.primitive.frontFace, "cw");
  assert.deepEqual(draw.args, [2, 1, 1, 0, 0]);
  assert.deepEqual([...new Uint16Array(draw.index.buffer.data)], [0, 1, 2, 0]);
  assert.equal(draw.uniform[19], 0.75);
  assert.equal(draw.uniform[20], Math.fround(0.3));
  assert.equal(draw.vertices.get(0), g.vertexBuffer);
  s.dispose();
  assert.equal(g.disposed, false);
});
test("sampled depth maps own bounded storage and reusable nearest comparison samplers", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 32, height: 16, maxDraws: 1 });
  assert.equal(s.allocatedBytes, 32 * 16 * 4 + 256);
  assert.equal(d.textures.length, 1);
  assert.equal(d.textures[0].format, "depth32float");
  assert.equal(d.textures[0].usage, 20);
  assert.throws(() => s.sample(d), errorCode("ANIMATION_SHADOW_UNRENDERED"));
  const vp = identity();
  s.render({ viewProjection: vp, draws: [] });
  const snapshot = s.sample(d);
  vp[12] = 12;
  assert.equal(snapshot.viewProjection[12], 0);
  assert.ok(Object.isFrozen(snapshot.viewProjection));
  assert.equal(snapshot.sampler.compare, "less-equal");
  assert.equal(snapshot.sampler.minFilter, "nearest");
  s.render({ viewProjection: identity(), draws: [] });
  assert.equal(s.version, 2);
  assert.equal(snapshot.version, 1);
  assert.equal(s.sample(d).view, snapshot.view);
  assert.equal(d.textures.length, 1);
  s.dispose();
  s.dispose();
  assert.equal(s.allocatedBytes, 0);
  assert.ok(d.textures[0].destroyed);
});
test("changed geometry versions and implicit world transforms invalidate stale shadows", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    g = gpu(),
    mesh = await s.addMesh(g);
  const render = () => s.render({ viewProjection: identity(), draws: [mesh] });
  render();
  s.sample(d);
  for (const change of [() => g.version++, () => g.poseVersion++, () => g.worldMatrix[12]++]) {
    change();
    assert.throws(() => s.sample(d), errorCode("ANIMATION_SHADOW_STALE"));
    render();
    s.sample(d);
  }
  s.render({ viewProjection: identity(), draws: [{ mesh, worldMatrix: identity() }] });
  g.worldMatrix[12]++;
  s.sample(d);
  g.disposed = true;
  assert.throws(() => s.sample(d), errorCode("ANIMATION_SHADOW_STALE"));
  s.dispose();
});
test("bad shadow inputs and final draws publish neither writes nor a new map version", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    mesh = await s.addMesh(gpu());
  for (const frame of [
    { viewProjection: identity(), draws: [{}] },
    { viewProjection: [1], draws: [] },
    { viewProjection: identity(), draws: [mesh, { mesh, count: 4 }] },
    { viewProjection: identity(), draws: [], colorView: {} },
  ]) {
    assert.throws(() => s.render(frame));
    assert.equal(d.submissions.length, 0);
    assert.equal(d.writes.length, 0);
    assert.equal(s.version, 0);
  }
  s.render({ viewProjection: identity(), draws: [mesh] });
  assert.throws(() => s.sample(deviceSpy()));
  mesh.dispose();
  assert.throws(() => s.render({ viewProjection: identity(), draws: [mesh] }));
  s.dispose();
});
test("unsupported lit/blended casters and invalid depth attachments fail explicitly", async () => {
  const d = deviceSpy(),
    r = await createGpuAnimationRenderer(d, { format: null, maxDraws: 1 });
  await assert.rejects(r.addMesh(gpu(), { shading: "lambert" }));
  await assert.rejects(r.addMesh(gpu(), { alphaMode: "BLEND" }));
  for (const frame of [
    { colorView: {}, depthView: {} },
    { depthView: {}, resolveTarget: {} },
    {},
  ]) {
    assert.throws(() => r.render({ ...frame, viewProjection: identity(), draws: [] }));
  }
  assert.equal(d.submissions.length, 0);
  assert.equal(d.buffers.length, 1);
  r.dispose();
  await assert.rejects(createGpuAnimationRenderer(d, { format: null, depthFormat: null }));
});
test("texture dimensions and combined GPU storage are checked before allocation", async () => {
  for (const options of [
    { width: 0 },
    { width: 8193 },
    { width: 16, maxBytes: 1024 },
    { width: 16, maxBytes: 1100, maxDraws: 1 },
    { unknown: true },
  ]) {
    const d = deviceSpy();
    await assert.rejects(createGpuAnimationShadowMap(d, options));
    assert.equal(d.buffers.length, 0);
    assert.equal(d.textures.length, 0);
  }
});
test("allocation errors release the depth map and renderer, not borrowed geometry", async () => {
  const d = deviceSpy(),
    create = d.createTexture;
  d.createTexture = (options) => {
    const t = create.call(d, options);
    d.scopeError = { message: "texture OOM" };
    return t;
  };
  await assert.rejects(createGpuAnimationShadowMap(d, { width: 8 }));
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.ok(d.textures.every((t) => t.destroyed));
  assert.equal(d.scopes.length, 0);
});
test("completion failure is cumulative and releases owned depth/buffer resources", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    g = gpu(),
    mesh = await s.addMesh(g);
  g.whenIdle = async () => {
    throw new Error("deformation failed");
  };
  s.render({ viewProjection: identity(), draws: [mesh] });
  await assert.rejects(s.whenIdle(), /deformation failed/);
  assert.ok(s.failed);
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.ok(d.textures[0].destroyed);
  assert.throws(() => s.sample(d));
  assert.equal(g.disposed, false);
  s.dispose();
});
test("device loss and reentrant getters cannot publish a shadow snapshot", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 });
  assert.throws(
    () =>
      s.render({
        get viewProjection() {
          s.dispose();
          return identity();
        },
        draws: [],
      }),
    errorCode("ANIMATION_SHADOW_REENTRANT"),
  );
  assert.equal(s.disposed, false);
  assert.equal(s.version, 0);
  d.lose({ message: "gone" });
  await tick();
  assert.ok(s.failed);
  assert.ok(d.textures[0].destroyed);
  assert.throws(() => s.render({ viewProjection: identity(), draws: [] }));
  s.dispose();
});
test("ordinary color pipelines, blend state, lighting and submission ordering are unchanged", async () => {
  const d = deviceSpy(),
    r = await createGpuAnimationRenderer(d, { maxDraws: 2 }),
    g = gpu();
  assert.equal(d.pipelines.length, 6);
  const mesh = await r.addMesh(g, { shading: "metallic-roughness", alphaMode: "BLEND" });
  const frame = {
    colorView: {},
    depthView: {},
    viewProjection: identity(),
    draws: [mesh],
    lighting: {
      cameraPosition: [0, 0, 5],
      lights: [{ type: "directional", direction: [0, 0, -1] }],
    },
  };
  r.render(frame);
  const draw = d.passes[0].draws[0];
  assert.equal(draw.pipeline.fragment.targets[0].format, "rgba8unorm");
  assert.equal(draw.pipeline.depthStencil.depthWriteEnabled, false);
  assert.match(draw.pipeline.fragment.module.code, /return vec4<f32>\(rgb/);
  g.worldMatrix[12] = 2;
  r.render(frame);
  assert.equal(draw.uniform[12], 0);
  assert.equal(d.passes[1].draws[0].uniform[12], 2);
  await r.whenIdle();
  r.dispose();
});

const lighting = () => ({
  cameraPosition: [0, 0, 5],
  lights: [
    { type: "directional", direction: [0, 0, -1] },
    { type: "point", position: [0, 1, 2] },
  ],
});
const colorFrame = (draws, shadow) => ({
  colorView: {},
  depthView: {},
  viewProjection: identity(),
  lighting: lighting(),
  draws,
  shadow,
});

test("shadow receiver binds depth comparison to only the selected direct light", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 32, height: 16 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  const g = gpu(),
    caster = await s.addMesh(g),
    receiver = await r.addMesh(g, { shading: "metallic-roughness", emissiveFactor: [0.2, 0.1, 0] });
  s.render({ viewProjection: identity(), draws: [caster] });
  r.render(colorFrame([receiver], { map: s, bias: 0.002, normalBias: 0.03, strength: 0.7 }));
  const draw = d.passes.at(-1).draws[0],
    group = draw.groups.get(1).group;
  assert.equal(group.entries[2].resource, s.sample(d).view);
  assert.equal(group.entries[3].resource.compare, "less-equal");
  const u = new Float32Array(group.entries[1].resource.buffer.data);
  assert.deepEqual([...u.slice(0, 16)], identity());
  assert.deepEqual(
    [...u.slice(16, 20)],
    [0, Math.fround(0.002), Math.fround(0.03), Math.fround(0.7)],
  );
  assert.deepEqual([...u.slice(20, 22)], [1 / 32, 1 / 16]);
  const code = draw.pipeline.fragment.module.code;
  assert.match(code, /if \(i == u32\(shadow_info.options.x\)\)/);
  assert.match(code, /var result = emission/);
  assert.match(code, /textureSampleCompareLevel/);
  assert.match(code, /0.5 - ndc.y \* 0.5/);
  assert.match(code, /reference = ndc.z - shadow_info.options.y/);
  assert.match(code, /visible \/ 9.0/);
  assert.match(code, /if \(clip.w <= 0.0\) \{ return 1.0/);
  assert.match(code, /\(nl \* attenuation\) \* visibility/);
  const allocated = d.buffers.length,
    groups = d.groups.length;
  r.render(colorFrame([receiver], { map: s }));
  assert.equal(d.buffers.length, allocated);
  assert.equal(d.groups.length, groups);
  r.render(colorFrame([receiver], null));
  assert.equal(d.passes.at(-1).draws[0].groups.get(1).group.entries.length, 1);
  assert.doesNotMatch(d.passes.at(-1).draws[0].pipeline.fragment.module.code, /shadow_depth/);
  await r.whenIdle();
  r.dispose();
  assert.equal(s.disposed, false);
  s.dispose();
});
test("all mapped material variants preserve their texture groups alongside shadows", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  s.render({ viewProjection: identity(), draws: [] });
  const texture = { view: {}, sampler: {} },
    mesh = await r.addMesh(gpu(), {
      shading: "metallic-roughness",
      texCoords: [0, 0, 1, 0, 0, 1],
      baseColorTexture: texture,
      metallicRoughnessTexture: texture,
      normalTexture: texture,
      emissiveTexture: texture,
      alphaMode: "MASK",
    });
  r.render(colorFrame([mesh], { map: s }));
  const draw = d.passes.at(-1).draws[0];
  assert.equal(draw.groups.get(1).group.entries.length, 8);
  assert.equal(draw.groups.get(2).group.entries.length, 4);
  assert.match(draw.pipeline.fragment.module.code, /@group\(2\) @binding\(2\) var shadow_depth/);
  assert.match(draw.pipeline.fragment.module.code, /dpdx\(input.world\)/);
  assert.match(draw.pipeline.fragment.module.code, /rgba.a < draw_info.options.x/);
  r.dispose();
  s.dispose();
});
test("directional and spot indices are supported but point/cube shadows are not guessed", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  const mesh = await r.addMesh(gpu(), { shading: "lambert" });
  s.render({ viewProjection: identity(), draws: [] });
  const f = colorFrame([mesh], { map: s, lightIndex: 1 });
  assert.throws(() => r.render(f), errorCode("ANIMATION_RENDER_SHADOW"));
  f.lighting.lights[1] = { type: "spot", position: [0, 0, 4], direction: [0, 0, -1] };
  r.render(f);
  const group = d.passes.at(-1).draws[0].groups.get(1).group;
  assert.equal(new Float32Array(group.entries[1].resource.buffer.data)[16], 1);
  r.dispose();
  s.dispose();
});
test("shadow options and stale/disposed maps fail before receiver GPU side effects", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  const g = gpu(),
    caster = await s.addMesh(g),
    mesh = await r.addMesh(g, { shading: "lambert" });
  assert.throws(
    () => r.render(colorFrame([mesh], { map: s })),
    errorCode("ANIMATION_SHADOW_UNRENDERED"),
  );
  s.render({ viewProjection: identity(), draws: [caster] });
  const counts = () => [d.writes.length, d.submissions.length, d.groups.length, r.version],
    before = counts();
  for (const shadow of [
    { map: s, bias: Infinity },
    { map: s, bias: 2 },
    { map: s, normalBias: -1 },
    { map: s, strength: 1.1 },
    { map: s, lightIndex: 2 },
    { map: s, unknown: 1 },
    {},
  ]) {
    assert.throws(() => r.render(colorFrame([mesh], shadow)));
    assert.deepEqual(counts(), before);
  }
  g.poseVersion++;
  assert.throws(
    () => r.render(colorFrame([mesh], { map: s })),
    errorCode("ANIMATION_SHADOW_STALE"),
  );
  assert.deepEqual(counts(), before);
  s.dispose();
  assert.throws(
    () => r.render(colorFrame([mesh], { map: s })),
    errorCode("ANIMATION_SHADOW_DISPOSED"),
  );
  r.dispose();
});
test("shadow receiver detects map redraws during draw getters and attachment feedback", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  const mesh = await r.addMesh(gpu(), { shading: "lambert" });
  s.render({ viewProjection: identity(), draws: [] });
  const frame = colorFrame([mesh], { map: s });
  frame.depthView = s.sample(d).view;
  assert.throws(() => r.render(frame), errorCode("ANIMATION_RENDER_SHADOW"));
  assert.equal(r.version, 0);
  assert.throws(
    () =>
      r.render(
        colorFrame(
          [
            {
              mesh,
              get worldMatrix() {
                s.render({ viewProjection: identity(), draws: [] });
                return identity();
              },
            },
          ],
          { map: s },
        ),
      ),
    errorCode("ANIMATION_RENDER_SHADOW"),
  );
  assert.equal(r.version, 0);
  r.dispose();
  s.dispose();
});
test("shadow resources, limits and pipeline variants remain opt-in and byte bounded", async () => {
  const d = deviceSpy(),
    disabled = await createGpuAnimationRenderer(d, { maxDraws: 1 });
  const mesh = await disabled.addMesh(gpu(), { shading: "lambert" });
  assert.equal(disabled.allocatedBytes, 256 + 544);
  assert.throws(
    () => disabled.render(colorFrame([mesh], { map: {} })),
    errorCode("ANIMATION_RENDER_SHADOW"),
  );
  disabled.dispose();
  const enabled = await createGpuAnimationRenderer(d, {
    shadows: true,
    maxDraws: 1,
    maxBytes: 256 + 544 + 96,
  });
  assert.equal(enabled.allocatedBytes, 256);
  await enabled.addMesh(gpu(), { shading: "lambert" });
  assert.equal(enabled.allocatedBytes, 256 + 544 + 96);
  enabled.dispose();
  const limited = await createGpuAnimationRenderer(d, {
    shadows: true,
    maxDraws: 1,
    maxBytes: 256 + 544 + 95,
  });
  await assert.rejects(
    limited.addMesh(gpu(), { shading: "lambert" }),
    errorCode("ANIMATION_RENDER_LIMIT"),
  );
  assert.equal(limited.allocatedBytes, 256);
  limited.dispose();
  d.limits.maxUniformBuffersPerShaderStage = 2;
  const capped = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  await assert.rejects(
    capped.addMesh(gpu(), { shading: "lambert" }),
    errorCode("ANIMATION_RENDER_LIMIT"),
  );
  capped.dispose();
});
test("shadow completion failures reach the receiving renderer without transferring ownership", async () => {
  const d = deviceSpy(),
    s = await createGpuAnimationShadowMap(d, { width: 8 }),
    r = await createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 });
  const casterGpu = gpu(),
    caster = await s.addMesh(casterGpu),
    mesh = await r.addMesh(gpu(), { shading: "lambert" });
  let reject;
  const pending = new Promise((_, b) => {
    reject = b;
  });
  casterGpu.whenIdle = () => pending;
  s.render({ viewProjection: identity(), draws: [caster] });
  r.render(colorFrame([mesh], { map: s }));
  reject(new Error("shadow caster compute failed"));
  await assert.rejects(r.whenIdle(), /shadow caster compute failed/);
  assert.ok(r.failed);
  assert.ok(s.failed);
  r.dispose();
  s.dispose();
});

// Use the production builder and complete shadow/render modules. Only unrelated
// pose decoding and scene/deformer creation are substituted at package assembly.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function packageFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-shadow-package-")),
    toolkit = path.join(root, "toolkit");
  fs.mkdirSync(toolkit);
  for (const name of [
    "build_animation.mjs",
    "animation_render.mjs",
    "animation_shadow.mjs",
    "animation_shadow_receiver.mjs",
    "animation_scene_shadow.mjs",
    "animation_shadow_view.mjs",
  ]) {
    fs.copyFileSync(new URL("./" + name, import.meta.url), path.join(toolkit, name));
  }
  fs.writeFileSync(
    path.join(toolkit, "animation_gltf.mjs"),
    "export function decodeGltfAnimation(model){return model;}",
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_runtime.mjs"),
    `export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}
export function createAnimationPlayer(){return {nodeCount:1,clips:[],instances:[],morphWeights:[],dispose(){}};}`,
  );
  for (const [file, name] of [
    ["animation_controller", "createAnimationController"],
    ["animation_deformer", "createAnimationDeformer"],
    ["animation_webgpu", "createGpuAnimationDeformer"],
    ["animation_scene", "createGpuAnimationScene"],
  ]) {
    fs.writeFileSync(
      path.join(toolkit, file + ".mjs"),
      `export function ${name}(){throw Error('unused package boundary');}`,
    );
  }
  for (const name of ["animation_draw_order.mjs", "animation_bounds.mjs"])
    fs.writeFileSync(path.join(toolkit, name), "export {};");
  const entry = path.join(root, "model.gltf");
  fs.writeFileSync(entry, JSON.stringify({ asset: { version: "2.0" }, nodes: [{}] }));
  const { buildAnimation } = await import(pathToFileURL(path.join(toolkit, "build_animation.mjs")));
  return { root, toolkit, entry, buildAnimation };
}
test("relocated GPU packages execute shadow casting and receiving without source modules", async () => {
  const f = await packageFixture(),
    out = path.join(f.root, "package"),
    built = f.buildAnimation(f.entry, out, { webgpu: true });
  for (const name of ["animation_shadow.mjs", "animation_shadow_receiver.mjs"]) {
    assert.ok(built.artifacts.some((x) => x.file === name));
    assert.deepEqual(
      fs.readFileSync(path.join(out, name)),
      fs.readFileSync(new URL("./" + name, import.meta.url)),
    );
  }
  const moved = path.join(f.root, "deployed");
  fs.renameSync(out, moved);
  fs.renameSync(f.toolkit, f.toolkit + ".unavailable");
  const api = await import(pathToFileURL(path.join(moved, built.gpuEntry))),
    d = deviceSpy(),
    g = gpu();
  const shadow = await api.createGpuAnimationShadowMap(d, { width: 8, maxDraws: 1 }),
    caster = await shadow.addMesh(g);
  const r = await api.createGpuAnimationRenderer(d, { shadows: true, maxDraws: 1 }),
    mesh = await r.addMesh(g, { shading: "lambert" });
  shadow.render({ viewProjection: identity(), draws: [caster] });
  r.render({
    colorView: {},
    depthView: {},
    viewProjection: identity(),
    draws: [mesh],
    lighting: { cameraPosition: [0, 0, 5], lights: [{ type: "directional" }] },
    shadow: { map: shadow },
  });
  await r.whenIdle();
  assert.equal(d.submissions.length, 2);
  assert.match(d.passes[1].draws[0].pipeline.label, /lit-shadow/);
  r.dispose();
  shadow.dispose();
});
test("GPU package budgets include shadow dependencies while CPU packages stay unchanged", async () => {
  const f = await packageFixture(),
    a = f.buildAnimation(f.entry, path.join(f.root, "sized"), { webgpu: true });
  const short = path.join(f.root, "short");
  assert.throws(
    () => f.buildAnimation(f.entry, short, { webgpu: true, maxBytes: a.outputBytes - 1 }),
    errorCode("GLTF_ANIMATION_LIMIT"),
  );
  assert.equal(fs.existsSync(short), false);
  assert.equal(
    f.buildAnimation(f.entry, path.join(f.root, "exact"), { webgpu: true, maxBytes: a.outputBytes })
      .outputBytes,
    a.outputBytes,
  );
  const cpu = f.buildAnimation(f.entry, path.join(f.root, "cpu"));
  assert.equal(cpu.emittedFiles.includes("animation_shadow.mjs"), false);
  assert.equal(cpu.emittedFiles.includes("animation_shadow_receiver.mjs"), false);
  const source = fs.readFileSync(path.join(f.toolkit, "build_animation.mjs"), "utf8");
  const before = source
    .replace(",'animation_shadow.mjs','animation_shadow_receiver.mjs'", "")
    .replace("export {createGpuAnimationShadowMap} from './animation_shadow.mjs';\\n", "");
  assert.notEqual(before, source);
  fs.writeFileSync(path.join(f.toolkit, "before.mjs"), before);
  const { buildAnimation: old } = await import(pathToFileURL(path.join(f.toolkit, "before.mjs")));
  const prior = old(f.entry, path.join(f.root, "prior"));
  assert.equal(prior.outputBytes, cpu.outputBytes);
  for (const name of cpu.emittedFiles)
    assert.deepEqual(
      fs.readFileSync(path.join(cpu.outDir, name)),
      fs.readFileSync(path.join(prior.outDir, name)),
    );
});
