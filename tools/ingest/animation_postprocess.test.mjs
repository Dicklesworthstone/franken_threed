import assert from "node:assert/strict";
import test from "node:test";
import { createGpuAnimationPostprocessor } from "./animation_postprocess.mjs";
import { deferred, gpuPostprocessFixture } from "./fixtures/animation/gpu_postprocess_fixture.mjs";

const code = (value) => ({ code: `ANIMATION_POSTPROCESS_${value}` });
async function setup(options = {}) {
  const f = gpuPostprocessFixture();
  f.post = await createGpuAnimationPostprocessor(f.device, {
    output: { format: "rgba8unorm" },
    bloom: { levels: 2 },
    ...options,
  });
  f.frame = { source: f.texture(), target: f.texture(16, 8, "rgba8unorm") };
  return f;
}
function sceneFixture() {
  const calls = [];
  return {
    calls,
    render(frame) {
      calls.push({ frame, method: "render" });
      return this;
    },
    renderCamera(frame, camera) {
      calls.push({ frame, camera, method: "renderCamera" });
      return this;
    },
    whenIdle() {
      return Promise.resolve();
    },
    update() {
      throw new Error("unexpected implicit update");
    },
    dispose() {
      throw new Error("borrowed scene disposed");
    },
  };
}

test("production bloom is submitted before the existing output pass using an owned linear intermediate", async () => {
  const f = await setup();
  assert.equal(f.post.render(f.frame), f.post);
  assert.equal(f.calls.submissions.length, 2);
  assert.equal(
    f.calls.submissions[0][0].passes.at(-1).pipeline.fragment.entryPoint,
    "composite_main",
  );
  assert.equal(f.calls.submissions[1][0].passes[0].pipeline.fragment.entryPoint, "fragment_main");
  const intermediate =
    f.calls.submissions[0][0].passes.at(-1).descriptor.colorAttachments[0].view.texture;
  assert.equal(intermediate.format, "rgba16float");
  assert.equal(
    f.calls.submissions[1][0].passes[0].bindings.entries[0].resource.texture,
    intermediate,
  );
  assert.equal(f.post.lastRender.postprocessSubmissions, 2);
  assert.equal(f.post.lastRender.output.toneMapping, "aces-filmic");
  assert.equal(f.post.allocatedBytes, 1696);
  await f.post.whenIdle();
  f.post.dispose();
});

test("zero-strength postprocessing skips bloom and HDR intermediate rather than submitting an identity bloom pass", async () => {
  const f = await setup({ maxBytes: 32, bloom: { strength: 0 } });
  f.post.render(f.frame);
  assert.equal(f.calls.textures.length, 0);
  assert.equal(f.calls.submissions.length, 1);
  assert.equal(f.post.allocatedBytes, 32);
  assert.equal(f.post.lastRender.bloom, null);
  assert.equal(
    f.calls.submissions[0][0].passes[0].bindings.entries[0].resource.texture,
    f.frame.source,
  );
  await f.post.whenIdle();
  f.post.dispose();
});

test("per-frame tone/exposure and bloom settings reach separate native uniform buffers without changing defaults", async () => {
  const f = await setup();
  f.post.render({
    ...f.frame,
    bloom: { strength: 2, threshold: 3, softKnee: 0.25 },
    output: { toneMapping: "reinhard", exposure: 0.5 },
  });
  const uniforms = f.calls.writes.map((write) => new DataView(write.bytes.buffer));
  assert.equal(uniforms[0].getFloat32(0, true), 3);
  assert.equal(uniforms[0].getFloat32(4, true), 0.75);
  assert.equal(uniforms[0].getFloat32(8, true), 2);
  assert.equal(uniforms[1].getFloat32(0, true), 0.5);
  assert.equal(uniforms[1].getUint32(4, true), 2);
  assert.equal(uniforms[1].getUint32(8, true), 0);
  assert.equal(uniforms[1].getUint32(12, true), 2);
  f.post.render(f.frame);
  assert.equal(f.post.lastRender.output.exposure, 1);
  assert.equal(f.post.lastRender.bloom.settings.strength, 1);
  await f.post.whenIdle();
  f.post.dispose();
});

test("managed scene rendering supplies HDR/depth attachments, preserves frame inputs, and does not advance animation", async () => {
  const f = await setup(),
    scene = sceneFixture(),
    matrix = Array(16).fill(0);
  f.post.renderScene(scene, {
    target: f.frame.target,
    frame: { viewProjection: matrix, lighting: { lights: [] } },
  });
  const draw = scene.calls[0];
  assert.equal(draw.method, "render");
  assert.equal(draw.frame.colorView.texture.format, "rgba16float");
  assert.equal(draw.frame.depthView.texture.format, "depth32float");
  assert.equal(draw.frame.viewProjection, matrix);
  assert.deepEqual(draw.frame.clearColor, [0, 0, 0, 1]);
  assert.equal(draw.frame.loadOp, "clear");
  assert.equal(f.post.lastRender.managedScene, true);
  const source = f.calls.submissions[0][0].passes[0].bindings.entries[0].resource.texture;
  assert.equal(source, draw.frame.colorView.texture);
  assert.equal(f.post.allocatedBytes, 3232);
  await f.post.whenIdle();
  f.post.dispose();
  assert.equal(f.frame.target.destroyed, 0);
  assert.equal(f.calls.deviceDestroyed, 0);
});

test("authored camera uses renderCamera and depth-free scenes omit the depth attachment", async () => {
  const f = await setup({ depthFormat: null }),
    scene = sceneFixture(),
    camera = { cameraNode: 3, aspectRatio: 2 };
  f.post.renderScene(scene, { target: f.frame.target, camera });
  assert.equal(scene.calls[0].method, "renderCamera");
  assert.equal(scene.calls[0].camera, camera);
  assert.equal(Object.hasOwn(scene.calls[0].frame, "depthView"), false);
  assert.ok(f.calls.textures.every((t) => t.format !== "depth32float"));
  await f.post.whenIdle();
  f.post.dispose();
});

test("same-size frames reuse managed targets and bloom storage; resized frames retire old targets", async () => {
  const f = await setup(),
    scene = sceneFixture();
  f.post.renderScene(scene, { target: f.frame.target });
  const initial = f.calls.textures.slice();
  f.post.renderScene(scene, { target: f.frame.target });
  assert.equal(f.calls.textures.length, initial.length);
  f.post.renderScene(scene, { target: f.texture(8, 8, "rgba8unorm") });
  assert.ok(initial.every((t) => t.destroyed === 1));
  assert.equal(f.post.lastRender.width, 8);
  await f.post.whenIdle();
  f.post.dispose();
  assert.ok(f.calls.textures.every((t) => t.destroyed === 1));
});

test("all managed attachment/color/camera requirements are checked before scene execution or texture allocation", async () => {
  const f = await setup(),
    scene = sceneFixture();
  const cases = [
    { frame: { colorView: {} } },
    { frame: { resolveTarget: {} } },
    { frame: { clearColor: [0, 0, 0, 0] } },
    { frame: { loadOp: "load" } },
    { frame: { depthLoadOp: "load" } },
    { output: { exposure: -1 } },
    { bloom: { strength: NaN } },
  ];
  for (const rest of cases)
    assert.throws(() => f.post.renderScene(scene, { target: f.frame.target, ...rest }));
  assert.throws(
    () =>
      f.post.renderScene(
        { render: scene.render, whenIdle: scene.whenIdle },
        { target: f.frame.target, camera: {} },
      ),
    code("SCENE"),
  );
  assert.equal(scene.calls.length, 0);
  assert.equal(f.calls.textures.length, 0);
  assert.equal(f.calls.submissions.length, 0);
  f.post.dispose();
});

test("invalid final target/settings are rejected before bloom submission and preserve prior render stats", async () => {
  const f = await setup();
  f.post.render(f.frame);
  const previous = f.post.lastRender,
    submissions = f.calls.submissions.length;
  for (const input of [
    { ...f.frame, target: f.texture(16, 8, "bgra8unorm") },
    { ...f.frame, target: f.texture(2, 2, "rgba8unorm") },
    { ...f.frame, output: { toneMapping: "bogus" } },
    { ...f.frame, output: { exposure: Infinity } },
    { ...f.frame, bloom: { softKnee: -1 } },
    { ...f.frame, output: { outputAlpha: "straight" } },
    { ...f.frame, source: { ...f.frame.source, sampleCount: 4 } },
  ])
    assert.throws(() => f.post.render(input));
  assert.equal(f.calls.submissions.length, submissions);
  assert.equal(f.post.lastRender, previous);
  assert.equal(f.post.failed, false);
  await f.post.whenIdle();
  f.post.dispose();
});

test("combined payload budget rejects allocations that would each fit independent stage budgets", async () => {
  const f = await setup({ maxBytes: 1500 });
  assert.throws(() => f.post.render(f.frame), code("LIMIT"));
  assert.equal(f.calls.textures.length, 0);
  assert.equal(f.calls.submissions.length, 0);
  f.post.render({ ...f.frame, bloom: { strength: 0 } });
  await f.post.whenIdle();
  f.post.dispose();
});

test("combined old/new resize overlap is checked before changing current resources", async () => {
  const f = await setup({ maxBytes: 2000, bloom: { levels: 1 } });
  const small = { source: f.texture(8, 8), target: f.texture(8, 8, "rgba8unorm") };
  f.post.render(small);
  assert.equal(f.post.allocatedBytes, 800);
  const count = f.calls.textures.length;
  assert.throws(() => f.post.render(f.frame), code("LIMIT"));
  assert.equal(f.calls.textures.length, count);
  assert.ok(f.calls.textures.every((t) => t.destroyed === 0));
  f.post.render(small);
  await f.post.whenIdle();
  f.post.dispose();
});

test("switching from externally supplied source to managed scene allocates only the missing target set", async () => {
  const f = await setup(),
    scene = sceneFixture();
  f.post.render(f.frame);
  const bloomTextures = f.calls.textures.slice(1);
  f.post.renderScene(scene, { target: f.frame.target });
  assert.ok(bloomTextures.every((t) => t.destroyed === 0));
  assert.equal(f.post.lastRender.managedScene, true);
  await f.post.whenIdle();
  f.post.dispose();
});

test("native allocation failure releases partially created targets and both existing child passes", async () => {
  const f = await setup(),
    scene = sceneFixture();
  f.controls.textureFailure = 2;
  assert.throws(() => f.post.renderScene(scene, { target: f.frame.target }), /allocation failure/);
  assert.equal(f.post.failed, true);
  assert.equal(f.post.allocatedBytes, 0);
  assert.equal(scene.calls.length, 0);
  assert.ok(f.calls.textures.every((t) => t.destroyed === 1));
  assert.ok(f.calls.buffers.every((b) => b.destroyed === 1));
  f.post.dispose();
});

test("failure of bloom initialization disposes an already initialized output pass", async () => {
  const f = gpuPostprocessFixture();
  let count = 0;
  const original = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = async (descriptor) => {
    if (++count > 1) throw new Error("bloom compilation failed");
    return original(descriptor);
  };
  await assert.rejects(createGpuAnimationPostprocessor(f.device), /bloom compilation failed/);
  assert.equal(f.calls.buffers.length, 1);
  assert.equal(f.calls.buffers[0].destroyed, 1);
});

test("borrowed scene completion failure is observed and releases composition resources without disposing the scene", async () => {
  const f = await setup(),
    scene = sceneFixture(),
    completion = deferred();
  scene.whenIdle = () => completion.promise;
  f.post.renderScene(scene, { target: f.frame.target });
  const idle = f.post.whenIdle();
  completion.reject(new Error("scene GPU failed"));
  await assert.rejects(idle, /scene GPU failed/);
  assert.equal(f.post.failed, true);
  assert.equal(f.post.allocatedBytes, 0);
  f.post.dispose();
});

test("device loss is terminal for both stages and all owned attachments", async () => {
  const f = await setup();
  f.post.render(f.frame);
  f.lost.resolve({ message: "lost during frame" });
  await assert.rejects(f.post.whenIdle(), /DEVICE_LOST|GPU/);
  assert.equal(f.post.allocatedBytes, 0);
  assert.ok(f.calls.textures.every((t) => t.destroyed === 1));
  assert.equal(f.frame.source.destroyed + f.frame.target.destroyed + f.calls.deviceDestroyed, 0);
  f.post.dispose();
});

test("disposal rejects pending frame completion and is idempotent", async () => {
  const f = await setup();
  f.controls.completion = deferred();
  f.post.render(f.frame);
  const idle = f.post.whenIdle();
  f.post.dispose();
  f.post.dispose();
  await assert.rejects(idle, code("DISPOSED"));
  assert.equal(f.post.allocatedBytes, 0);
  assert.ok(f.calls.buffers.every((b) => b.destroyed === 1));
  assert.throws(() => f.post.render(f.frame), code("DISPOSED"));
});

test("frame getters and borrowed scene calls cannot reenter or dispose the composition", async () => {
  const f = await setup();
  assert.throws(
    () =>
      f.post.render({
        get source() {
          f.post.dispose();
          return f.frame.source;
        },
        target: f.frame.target,
      }),
    code("REENTRANT"),
  );
  const scene = sceneFixture();
  scene.render = () => f.post.render(f.frame);
  assert.throws(() => f.post.renderScene(scene, { target: f.frame.target }), code("REENTRANT"));
  assert.equal(f.post.disposed, false);
  assert.equal(f.calls.submissions.length, 0);
  f.post.dispose();
});

test("unknown configuration and unsupported depth/alpha policies fail before native construction", async () => {
  for (const options of [
    { wrong: 1 },
    { depthFormat: "depth24plus" },
    { maxBytes: 31 },
    { bloom: { wrong: true } },
    { output: { inputAlpha: "premultiplied" } },
  ]) {
    const f = gpuPostprocessFixture();
    await assert.rejects(createGpuAnimationPostprocessor(f.device, options));
    assert.equal(f.calls.modules.length, 0);
  }
});

test("native pixel fixture refuses to label a host spy as real WebGPU execution", async () => {
  const { runAnimationBloomBrowserChecks } = await import("./animation_bloom.browser.mjs");
  await assert.rejects(
    runAnimationBloomBrowserChecks(gpuPostprocessFixture().device),
    /NATIVE_DEVICE_REQUIRED/,
  );
});
