import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { createGpuAnimationScene } from "./animation_scene.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const code = (expected) => (error) => error.code === expected;
function pose() {
  return createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [
      { translation: [5, 0, 0], weights: [0] },
      { translation: [-5, 0, 0], weights: [0] },
      {},
    ],
    skins: [{ joints: [2] }],
    instances: [
      { node: 0, skin: 0 },
      { node: 1, skin: 0 },
    ],
    clips: [
      {
        channels: [
          { node: 2, path: "translation", times: [0, 2], values: [0, 0, 0, 2, 2, 0] },
          { node: 0, path: "weights", times: [0, 2], values: [0, 1] },
          { node: 1, path: "weights", times: [0, 2], values: [0, 1] },
        ],
      },
    ],
  });
}
function drawables() {
  return [0, 1].map((node) => ({
    geometry: {
      node,
      positions: [-0.2, -0.2, 0, 0.2, -0.2, 0, 0, 0.2, 0],
      joints: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      weights: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
      morphTargets: [{ positions: [0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0] }],
    },
    indices: [0, 1, 2],
    baseColor: node ? [0, 0, 1, 1] : [1, 0, 0, 1],
  }));
}
// A recording device, not GPU execution. Real pose/controller/deformation/render
// modules execute; only WebGPU calls are recorded, never software-emulated here.
function deviceSpy() {
  const buffers = [],
    submissions = [],
    writes = [],
    scopes = [];
  let resolveLoss;
  const d = {
    buffers,
    submissions,
    writes,
    scopes,
    lost: new Promise((resolve) => {
      resolveLoss = resolve;
    }),
    lose: () => resolveLoss({ message: "lost" }),
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 1 << 28,
      maxUniformBufferBindingSize: 65536,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8,
      maxVertexAttributes: 16,
      maxVertexBufferArrayStride: 2048,
      maxStorageBuffersPerShaderStage: 8,
      maxBindingsPerBindGroup: 1000,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 1 << 27,
    },
    pushErrorScope(type) {
      scopes.push(type);
    },
    popErrorScope() {
      assert.ok(scopes.pop());
      return Promise.resolve(null);
    },
    createBuffer(options) {
      const data = new ArrayBuffer(options.size),
        b = {
          ...options,
          data,
          destroyed: false,
          getMappedRange: () => data,
          unmap() {},
          destroy() {
            b.destroyed = true;
          },
        };
      buffers.push(b);
      return b;
    },
    createShaderModule: (options) => options,
    createBindGroupLayout: (options) => options,
    createPipelineLayout: (options) => options,
    createBindGroup: (options) => options,
    createRenderPipelineAsync: (options) => Promise.resolve(options),
    createComputePipelineAsync: (options) =>
      Promise.resolve({ ...options, getBindGroupLayout: () => ({}) }),
    createCommandEncoder() {
      const passes = [];
      function pass(kind, descriptor) {
        const p = { kind, descriptor, draws: [] };
        passes.push(p);
        let group,
          vertex,
          pipeline,
          offset = 0;
        return {
          setPipeline(v) {
            pipeline = v;
          },
          setBindGroup(slot, value, offsets = [0]) {
            group = value;
            offset = offsets[0];
          },
          setVertexBuffer(slot, value) {
            vertex = value;
          },
          setIndexBuffer() {},
          setViewport() {},
          setScissorRect() {},
          dispatchWorkgroups() {},
          draw(...args) {
            p.draws.push({ group, offset, vertex, pipeline, args });
          },
          drawIndexed(...args) {
            p.draws.push({ group, offset, vertex, pipeline, args });
          },
          end() {},
        };
      }
      return {
        beginComputePass: (d) => pass("compute", d),
        beginRenderPass: (d) => pass("render", d),
        finish: () => passes,
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, start = 0, size = data.length - start) {
        const n = data.BYTES_PER_ELEMENT ?? 1,
          bytes = new Uint8Array(data.buffer, data.byteOffset + start * n, size * n).slice();
        new Uint8Array(buffer.data, offset, bytes.length).set(bytes);
        writes.push({ buffer, bytes });
      },
      submit(commands) {
        for (const passes of commands)
          for (const p of passes) {
            for (const draw of p.draws)
              draw.uniforms = new Float32Array(
                draw.group.entries[0].resource.buffer.data,
                draw.offset,
                24,
              ).slice();
            submissions.push(p);
          }
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };
  return d;
}
const frame = () => ({ colorView: {}, depthView: {}, viewProjection: identity() });

test("real controller advances once while every mesh uploads before the render pass", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables(), { maxBytes: 2000 });
  const allocationCount = d.buffers.length,
    vertices = scene.deformers.map((gpu) => gpu.vertexBuffer);
  assert.equal(
    scene.bufferBytes,
    d.buffers.reduce((n, b) => n + b.size, 0),
  );
  assert.equal(scene.bufferBytes, 1584);
  const action = scene.controller.createAction(0, { loop: "once", clampWhenFinished: true }).play();
  d.submissions.length = 0;
  for (let i = 0; i < 60; i++) {
    scene.update(1 / 30);
    scene.render(frame());
  }
  await scene.whenIdle();
  assert.equal(action.finished, true);
  assert.ok(Math.abs(scene.controller.time - 2) < 1e-12);
  assert.equal(d.buffers.length, allocationCount);
  assert.equal(d.submissions.length, 180);
  for (let i = 0; i < 60; i++)
    assert.deepEqual(
      d.submissions.slice(i * 3, i * 3 + 3).map((p) => p.kind),
      ["compute", "compute", "render"],
    );
  const last = d.submissions.at(-1);
  assert.equal(last.draws[0].vertex, vertices[0]);
  assert.equal(last.draws[1].vertex, vertices[1]);
  assert.equal(last.draws[0].uniforms[12], 5);
  assert.equal(last.draws[1].uniforms[12], -5);
  assert.deepEqual([...last.draws[0].uniforms.slice(16, 20)], [1, 0, 0, 1]);
  assert.deepEqual([...last.draws[1].uniforms.slice(16, 20)], [0, 0, 1, 1]);
  const palettes = d.writes
    .filter((w) => w.buffer.label.endsWith("/3"))
    .slice(-2)
    .map((w) => new Float32Array(w.bytes.buffer));
  assert.equal(palettes[0][12], -3);
  assert.equal(palettes[1][12], 7);
  assert.equal(scene.poseVersion, p.version);
  assert.ok(scene.deformers.every((g) => g.poseVersion === p.version));
  scene.dispose();
  assert.equal(scene.bufferBytes, 0);
  assert.equal(p.disposed, false);
});

test("direct pose changes require an explicit complete upload; bad CPU updates preserve last frame", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables());
  p.sample(1);
  const before = d.submissions.length;
  assert.throws(() => scene.render(frame()), code("ANIMATION_SCENE_STALE"));
  assert.equal(d.submissions.length, before);
  scene.upload();
  scene.render(frame());
  const version = p.version,
    submitted = d.submissions.length;
  assert.throws(() => scene.update(NaN));
  assert.equal(p.version, version);
  assert.equal(d.submissions.length, submitted);
  assert.equal(scene.failed, false);
  scene.render(frame());
  await scene.whenIdle();
  scene.dispose();
});

test("failure on a later GPU submission prevents any partial frame and releases the group", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables());
  scene.controller.createAction(0).play();
  const submit = d.queue.submit.bind(d.queue),
    error = new Error("second mesh failed");
  let compute = 0;
  d.queue.submit = (commands) => {
    if (commands[0][0].kind === "compute" && ++compute === 2) throw error;
    submit(commands);
  };
  const before = d.submissions.length;
  assert.throws(
    () => scene.update(0.5),
    (e) => e === error,
  );
  assert.equal(scene.failed, true);
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.equal(scene.bufferBytes, 0);
  assert.equal(p.disposed, false);
  assert.equal(scene.controller.time, 0.5, "Submitted CPU time is not falsely rolled back");
  assert.deepEqual(
    d.submissions.slice(before).map((p) => p.kind),
    ["compute"],
  );
  assert.throws(
    () => scene.render(frame()),
    (e) => e === error,
  );
  scene.dispose();
  scene.dispose();
});

test("asynchronous validation failures propagate through scene completion and free all children", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables()),
    pop = d.popErrorScope.bind(d);
  let first = true;
  d.popErrorScope = () => {
    const promise = pop();
    if (first) {
      first = false;
      return Promise.resolve({ message: "invalid compute" });
    }
    return promise;
  };
  scene.update(0.5);
  await assert.rejects(scene.whenIdle(), code("ANIMATION_GPU_DEVICE"));
  assert.equal(scene.failed, true);
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.equal(p.disposed, false);
  scene.dispose();
});

test("bad later geometry and aggregate budget failures unwind successful earlier meshes", async () => {
  for (const change of [
    (items) => {
      items[1].indices = [0, 1, 999];
    },
    (items) => {
      items[1].geometry.weights[0] = 0.5;
    },
  ]) {
    const p = pose(),
      d = deviceSpy(),
      items = drawables();
    change(items);
    await assert.rejects(createGpuAnimationScene(d, p, items));
    assert.ok(d.buffers.every((b) => b.destroyed));
    assert.equal(p.disposed, false);
    assert.equal(p.version, 0);
  }
  const p = pose(),
    d = deviceSpy();
  await assert.rejects(
    createGpuAnimationScene(d, p, drawables(), { maxBytes: 1200 }),
    code("ANIMATION_GPU_LIMIT"),
  );
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.ok(d.buffers.reduce((sum, b) => sum + b.size, 0) <= 1200);
});

test("pose mutation during asynchronous initialization rejects mixed-version resources", async () => {
  const p = pose(),
    d = deviceSpy(),
    create = d.createComputePipelineAsync;
  let first = true;
  d.createComputePipelineAsync = (options) =>
    create(options).then((pipeline) => {
      if (first) {
        first = false;
        p.sample(1);
      }
      return pipeline;
    });
  await assert.rejects(createGpuAnimationScene(d, p, drawables()), code("ANIMATION_SCENE_CHANGED"));
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.equal(p.disposed, false);
});

test("invalid render arguments can be corrected without advancing playback twice", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables());
  scene.controller.createAction(0).play();
  scene.update(0.25);
  const version = p.version,
    time = scene.controller.time;
  assert.throws(
    () => scene.render({ ...frame(), depthView: null }),
    code("ANIMATION_RENDER_ATTACHMENT"),
  );
  assert.equal(scene.failed, false);
  scene.render(frame());
  assert.equal(p.version, version);
  assert.equal(scene.controller.time, time);
  await scene.whenIdle();
  scene.dispose();
});

test("caller can select draw order without surrendering borrowed pose/device ownership", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables());
  scene.render({ ...frame(), draws: [scene.draws[1], scene.draws[0]] });
  const pass = d.submissions.at(-1);
  assert.equal(pass.draws[0].vertex, scene.deformers[1].vertexBuffer);
  assert.equal(pass.draws[1].vertex, scene.deformers[0].vertexBuffer);
  scene.dispose();
  assert.equal(scene.controller.disposed, true);
  assert.ok(scene.deformers.every((g) => g.disposed));
  assert.equal(p.disposed, false);
  p.sample(1);
  assert.throws(() => scene.upload(), code("ANIMATION_SCENE_DISPOSED"));
  await assert.rejects(scene.whenIdle(), code("ANIMATION_SCENE_DISPOSED"));
});

test("device loss and reentrant disposal cannot leave a drawable partial scene", async () => {
  const p = pose(),
    d = deviceSpy(),
    scene = await createGpuAnimationScene(d, p, drawables());
  assert.throws(
    () =>
      scene.update(0, {
        get rootMatrix() {
          scene.dispose();
          return identity();
        },
      }),
    code("ANIMATION_SCENE_REENTRANT"),
  );
  assert.equal(scene.disposed, false);
  d.lose();
  await assert.rejects(scene.whenIdle());
  assert.equal(scene.failed, true);
  assert.ok(d.buffers.every((b) => b.destroyed));
  scene.dispose();
});
