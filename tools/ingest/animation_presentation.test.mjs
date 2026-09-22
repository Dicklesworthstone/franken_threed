import assert from "node:assert/strict";
import test from "node:test";
import { createGpuAnimationPresentation } from "./animation_presentation.mjs";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((r, j) => {
    resolve = r;
    reject = j;
  });
  return { promise, resolve, reject };
};
function setup() {
  const textures = [],
    buffers = [],
    events = [],
    scopes = [],
    lost = deferred();
  let textureFailure = 0;
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    lost: lost.promise,
    queue: {
      writeBuffer(b, o, bytes) {
        events.push(["parameters", new Uint8Array(bytes).slice()]);
      },
      submit() {
        events.push(["output-submit"]);
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
    createShaderModule: (x) => x,
    createBindGroupLayout: (x) => x,
    createPipelineLayout: (x) => x,
    createRenderPipelineAsync: async (x) => x,
    createBindGroup: (x) => x,
    createBuffer(x) {
      const b = {
        ...x,
        destroyed: 0,
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    pushErrorScope() {},
    popErrorScope: () => scopes.shift() ?? Promise.resolve(null),
    createTexture(x) {
      if (textureFailure && textures.length + 1 === textureFailure) throw new Error("allocation");
      const t = texture(x.format, ...x.size, x.sampleCount, x.usage);
      textures.push(t);
      return t;
    },
    createCommandEncoder: () => ({
      beginRenderPass: (desc) => ({
        setPipeline() {},
        setBindGroup() {},
        draw() {
          events.push(["output-draw", desc]);
        },
        end() {},
      }),
      finish: () => ({}),
    }),
  };
  function texture(
    format = "bgra8unorm",
    width = 8,
    height = 4,
    depthOrArrayLayers = 1,
    sampleCount = 1,
    usage = 16,
  ) {
    return {
      format,
      width,
      height,
      depthOrArrayLayers,
      sampleCount,
      usage,
      dimension: "2d",
      destroyed: 0,
      createView: function (desc = {}) {
        return { texture: this, desc };
      },
      destroy() {
        this.destroyed++;
      },
    };
  }
  const scene = {
    format: "rgba16float",
    sampleCount: 1,
    depthFormat: "depth24plus",
    failed: false,
    disposed: false,
    calls: [],
    render(frame) {
      this.calls.push(frame);
      events.push(["scene-submit", frame]);
      return this;
    },
    renderCamera(frame, settings) {
      this.cameraSettings = settings;
      return this.render(frame);
    },
  };
  return {
    device,
    textures,
    buffers,
    events,
    scopes,
    lost,
    scene,
    target: () => texture(),
    texture,
    set textureFailure(n) {
      textureFailure = n;
    },
  };
}
for (const sampleCount of [1, 4])
  test(`${sampleCount}x scene resolves linear HDR before display conversion with matching depth`, async () => {
    const g = setup(),
      p = await createGpuAnimationPresentation(g.device, { sampleCount, toneMapping: "agx" });
    g.scene.sampleCount = sampleCount;
    const target = g.target();
    p.render(g.scene, { target, viewProjection: "borrowed", lighting: "lights", draws: [] });
    await p.whenIdle();
    const frame = g.scene.calls[0];
    assert.equal(frame.colorView.texture.format, "rgba16float");
    assert.equal(frame.colorView.texture.sampleCount, sampleCount);
    assert.equal(frame.depthView.texture.sampleCount, sampleCount);
    assert.equal(frame.depthView.texture.format, "depth24plus");
    assert.equal(Boolean(frame.resolveTarget), sampleCount === 4);
    if (sampleCount === 4) assert.equal(frame.resolveTarget.texture.sampleCount, 1);
    assert.deepEqual(
      g.events.filter((e) => e[0].endsWith("submit")).map((e) => e[0]),
      ["scene-submit", "output-submit"],
    );
    assert.equal(p.textureBytes, 8 * 4 * (8 * (sampleCount === 4 ? 5 : 1) + 4 * sampleCount));
    assert.equal(p.bufferBytes, 16);
    assert.equal(p.version, 1);
    assert.deepEqual([p.width, p.height], [8, 4]);
    assert.equal(frame.target, undefined);
    assert.equal(frame.output, undefined);
    p.dispose();
    assert.ok(g.textures.every((t) => t.destroyed === 1));
    assert.equal(target.destroyed, 0);
    assert.equal(g.scene.disposed, false);
  });
test("depthless scene allocates only color and forwards draw/camera/viewport parameters", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device, { depthFormat: null });
  g.scene.depthFormat = null;
  const camera = { cameraNode: 2, aspectRatio: 2 },
    viewport = [0, 0, 8, 4, 0, 1],
    draws = [{}, {}];
  p.renderCamera(g.scene, { target: g.target(), viewport, draws }, camera);
  assert.equal(g.scene.cameraSettings, camera);
  assert.equal(g.scene.calls[0].depthView, undefined);
  assert.equal(g.scene.calls[0].viewport, viewport);
  assert.equal(g.scene.calls[0].draws, draws);
  assert.equal(g.textures.length, 1);
  p.dispose();
});
test("clear color is premultiplied before scene blending; display options affect only the output pass", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device),
    clear = Object.freeze([4, 2, 1, 0.25]);
  const frame = Object.freeze({
    target: g.target(),
    clearColor: clear,
    output: Object.freeze({ toneMapping: "reinhard", exposure: 2, outputAlpha: "straight" }),
  });
  p.render(g.scene, frame);
  assert.deepEqual(g.scene.calls[0].clearColor, [1, 0.5, 0.25, 0.25]);
  assert.deepEqual(clear, [4, 2, 1, 0.25]);
  const data = new DataView(g.events.find((e) => e[0] === "parameters")[1].buffer);
  assert.deepEqual(
    [
      data.getFloat32(0, true),
      data.getUint32(4, true),
      data.getUint32(8, true),
      data.getUint32(12, true),
    ],
    [2, 2, 1, 0],
  );
  p.dispose();
});
test("same-size display targets reuse intermediate storage and preserve explicitly loaded HDR history", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  p.render(g.scene, { target: g.target() });
  const allocated = g.textures.length,
    first = g.scene.calls[0];
  p.render(g.scene, { target: g.target(), loadOp: "load", depthLoadOp: "load" });
  assert.equal(g.textures.length, allocated);
  assert.equal(g.scene.calls[1].colorView, first.colorView);
  assert.equal(g.scene.calls[1].depthView, first.depthView);
  assert.equal(p.version, 2);
  p.dispose();
});
test("resize matches new destination dimensions and retires old storage only after both submissions", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  p.render(g.scene, { target: g.target() });
  const old = [...g.textures];
  const original = g.scene.render;
  g.scene.render = function (frame) {
    assert.ok(old.every((t) => t.destroyed === 0));
    return original.call(this, frame);
  };
  p.render(g.scene, { target: g.texture("bgra8unorm", 10, 6) });
  assert.ok(old.every((t) => t.destroyed === 1));
  assert.equal(p.textureBytes, 10 * 6 * 12);
  assert.deepEqual([p.width, p.height], [10, 6]);
  assert.equal(p.version, 2);
  p.dispose();
});
test("initial/resize load refuses uninitialized history without allocating or changing previous contents", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  assert.throws(() => p.render(g.scene, { target: g.target(), loadOp: "load" }), {
    code: "ANIMATION_PRESENTATION_HISTORY",
  });
  assert.equal(g.textures.length, 0);
  p.render(g.scene, { target: g.target() });
  const prior = g.scene.calls[0];
  for (const mode of ["loadOp", "depthLoadOp"])
    assert.throws(
      () => p.render(g.scene, { target: g.texture("bgra8unorm", 9, 4), [mode]: "load" }),
      { code: "ANIMATION_PRESENTATION_HISTORY" },
    );
  assert.equal(g.scene.calls.length, 1);
  assert.equal(prior.colorView.texture.destroyed, 0);
  assert.equal(p.version, 1);
  p.dispose();
});
test("resize peak budget includes old plus new attachments, without destroying useful history", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device, { maxTextureBytes: 8 * 4 * 12 });
  p.render(g.scene, { target: g.target() });
  assert.throws(() => p.render(g.scene, { target: g.texture("bgra8unorm", 4, 4) }), {
    code: "ANIMATION_PRESENTATION_LIMIT",
  });
  assert.equal(p.version, 1);
  assert.equal(g.textures.length, 2);
  assert.ok(g.textures.every((t) => t.destroyed === 0));
  p.dispose();
});
test("bad output settings, feedback attachments, extents, destination formats and scene profiles fail before scene submission", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  const target = g.target();
  for (const frame of [
    { target, output: { exposure: -1 } },
    { target, output: { toneMapping: "custom" } },
    { target, output: { inputAlpha: "straight" } },
    { target, colorView: {} },
    { target, depthView: {} },
    { target, resolveTarget: {} },
    { target: g.texture("rgba16float") },
    { target: g.texture("bgra8unorm", 8193, 4) },
    { target: g.texture("bgra8unorm", 8, 4, 1, 4) },
    { target, clearColor: [1, 1, 1, 2] },
  ])
    assert.throws(() => p.render(g.scene, frame));
  g.scene.format = "rgba8unorm";
  assert.throws(() => p.render(g.scene, { target }), { code: "ANIMATION_PRESENTATION_FORMAT" });
  assert.equal(g.textures.length, 0);
  assert.equal(g.scene.calls.length, 0);
  assert.equal(p.failed, false);
  p.dispose();
});
test("a recoverable scene validation error during resize preserves the last submitted history", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  p.render(g.scene, { target: g.target() });
  const old = [...g.textures],
    original = g.scene.render;
  g.scene.render = () => {
    throw new Error("invalid camera");
  };
  assert.throws(
    () => p.render(g.scene, { target: g.texture("bgra8unorm", 4, 4) }),
    /invalid camera/,
  );
  assert.ok(old.every((t) => t.destroyed === 0));
  assert.ok(g.textures.slice(2).every((t) => t.destroyed === 1));
  assert.equal(p.failed, false);
  assert.equal(p.width, 8);
  g.scene.render = original;
  p.render(g.scene, { target: g.target(), loadOp: "load" });
  assert.equal(p.version, 2);
  p.dispose();
});
test("partial allocation failure cleans every created object and is terminal", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device, { sampleCount: 4 });
  g.scene.sampleCount = 4;
  g.textureFailure = 2;
  assert.throws(() => p.render(g.scene, { target: g.target() }), /allocation/);
  assert.equal(p.failed, true);
  assert.equal(p.textureBytes, 0);
  assert.equal(g.textures.length, 1);
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(g.buffers[0].destroyed, 1);
  p.dispose();
});
test("deferred texture allocation validation is included in whenIdle and releases targets", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device),
    bad = deferred();
  g.scopes.push(bad.promise, Promise.resolve(null));
  p.render(g.scene, { target: g.target() });
  bad.resolve({ message: "bad texture" });
  await assert.rejects(p.whenIdle(), { code: "ANIMATION_PRESENTATION_GPU" });
  assert.equal(p.textureBytes, 0);
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  p.dispose();
});
test("device loss releases HDR/MSAA/depth/output resources without destroying borrowed objects", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  const target = g.target();
  p.render(g.scene, { target });
  g.lost.resolve({ message: "lost" });
  await assert.rejects(p.whenIdle());
  assert.equal(p.failed, true);
  assert.equal(p.textureBytes, 0);
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  assert.equal(g.buffers[0].destroyed, 1);
  assert.equal(target.destroyed, 0);
  assert.equal(g.scene.disposed, false);
  p.dispose();
});
test("foreign asynchronous render is not accepted as a successful submitted scene", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  g.scene.render = () => Promise.reject(new Error("foreign async error"));
  assert.throws(() => p.render(g.scene, { target: g.target() }), {
    code: "ANIMATION_PRESENTATION_ASYNC",
  });
  assert.equal(p.failed, true);
  assert.equal(p.version, 0);
  assert.ok(g.textures.every((t) => t.destroyed === 1));
  p.dispose();
});
test("reentrant disposal is rejected and final disposal is idempotent", async () => {
  const g = setup(),
    p = await createGpuAnimationPresentation(g.device);
  g.scene.render = () => p.dispose();
  assert.throws(() => p.render(g.scene, { target: g.target() }), {
    code: "ANIMATION_PRESENTATION_REENTRANT",
  });
  assert.equal(p.disposed, false);
  p.dispose();
  p.dispose();
  assert.equal(g.buffers[0].destroyed, 1);
  assert.ok(g.textures.every((t) => t.destroyed === 1));
});
for (const option of [
  { sampleCount: 2 },
  { depthFormat: "depth24plus-stencil8" },
  { maxTextureBytes: 0 },
  { inputAlpha: "straight" },
])
  test("invalid presentation profile " + JSON.stringify(option), async () => {
    const g = setup();
    await assert.rejects(createGpuAnimationPresentation(g.device, option));
    assert.equal(g.buffers.length, 0);
    assert.equal(g.textures.length, 0);
  });
