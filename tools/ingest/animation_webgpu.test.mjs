import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { ANIMATION_DEFORM_WGSL, createGpuAnimationDeformer } from "./animation_webgpu.mjs";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((r, j) => {
    resolve = r;
    reject = j;
  });
  return { promise, resolve, reject };
};
const code = (expected) => (error) => error.code === expected;
// A descriptor/lifetime/queue spy, NOT a GPU or a WGSL interpreter. Actual
// shader execution and rendering are exercised by animation_webgpu.browser.mjs.
function deviceSpy() {
  const loss = deferred(),
    buffers = [],
    scopes = [],
    submissions = [],
    writes = [];
  const device = {
    limits: {
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBuffersPerShaderStage: 12,
      maxBindingsPerBindGroup: 1000,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65535,
      maxBufferSize: 268435456,
      maxUniformBufferBindingSize: 65536,
      maxStorageBufferBindingSize: 134217728,
    },
    lost: loss.promise,
    buffers,
    scopes,
    submissions,
    writes,
    loss,
    pushErrorScope(type) {
      scopes.push(type);
    },
    popErrorScope() {
      assert.ok(scopes.pop());
      const result = device.scopeError;
      device.scopeError = null;
      return Promise.resolve(result ?? null);
    },
    createBuffer(descriptor) {
      const buffer = {
        ...descriptor,
        bytes: new ArrayBuffer(descriptor.size),
        destroyed: 0,
        mapped: descriptor.mappedAtCreation,
        getMappedRange() {
          assert.equal(this.mapped, true);
          return this.bytes;
        },
        unmap() {
          assert.equal(this.mapped, true);
          this.mapped = false;
        },
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule(descriptor) {
      assert.equal(descriptor.code, ANIMATION_DEFORM_WGSL);
      return descriptor;
    },
    async createComputePipelineAsync(descriptor) {
      assert.equal(descriptor.compute.entryPoint, "deform");
      if (device.pipelineGate) await device.pipelineGate.promise;
      return {
        getBindGroupLayout(index) {
          assert.equal(index, 0);
          return {};
        },
      };
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createCommandEncoder() {
      if (device.encoderError) throw device.encoderError;
      let bound,
        dispatch,
        ended = false;
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup(index, group) {
              assert.equal(index, 0);
              bound = group;
            },
            dispatchWorkgroups(...dimensions) {
              dispatch = dimensions;
            },
            end() {
              ended = true;
            },
          };
        },
        finish() {
          assert.equal(ended, true);
          return { bound, dispatch };
        },
      };
    },
    queue: {
      writeBuffer(buffer, offset, array) {
        const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice();
        new Uint8Array(buffer.bytes).set(bytes, offset);
        writes.push({ buffer, offset, bytes });
      },
      submit(commands) {
        for (const command of commands)
          submissions.push({
            dispatch: command.dispatch,
            palette: new Float32Array(command.bound.entries[3].resource.buffer.bytes).slice(),
            weights: new Float32Array(command.bound.entries[4].resource.buffer.bytes).slice(),
          });
      },
      onSubmittedWorkDone() {
        return device.completionGate?.promise ?? Promise.resolve();
      },
    },
  };
  return device;
}
function scene() {
  return createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [
      { translation: [5, 0, 0], weights: [0] },
      { translation: [2, 0, 0], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] },
      { translation: [-7, 0, 0], weights: [0] },
    ],
    skins: [{ joints: [1] }],
    instances: [
      { node: 0, skin: 0 },
      { node: 2, skin: 0 },
    ],
    clips: [
      {
        channels: [
          { node: 1, path: "translation", times: [0, 2], values: [2, 0, 0, 4, 0, 0] },
          { node: 0, path: "weights", times: [0, 2], values: [0, 1] },
          { node: 2, path: "weights", times: [0, 2], values: [0, 1] },
        ],
      },
    ],
  });
}
function geometry(node = 0) {
  return {
    node,
    positions: [1, 0, 0],
    normals: [1, 0, 0],
    tangents: [1, 0, 0, -1],
    morphTargets: [{ positions: [2, 0, 0], normals: [0, 1, 0], tangents: [0, 2, 0] }],
    joints: [0, 0, 0, 0],
    weights: [1, 0, 0, 0],
  };
}

test("packs GPU base, morph, influence and config layouts without changing geometry", async () => {
  const device = deviceSpy(),
    pose = scene(),
    source = geometry(),
    before = structuredClone(source);
  const gpu = await createGpuAnimationDeformer(device, pose, source);
  assert.deepEqual(source, before);
  assert.equal(device.buffers.length, 7);
  assert.deepEqual([...new Float32Array(device.buffers[0].bytes)], [1, 0, 0, 1, 0, 0, 1, 0, 0, -1]);
  assert.deepEqual([...new Float32Array(device.buffers[1].bytes)], [2, 0, 0, 0, 1, 0, 0, 2, 0]);
  assert.deepEqual(
    [...new Uint32Array(device.buffers[2].bytes)],
    [0, 1065353216, 0, 0, 0, 0, 0, 0],
  );
  assert.deepEqual([...new Uint32Array(device.buffers[6].bytes)], [1, 1, 4, 1]);
  assert.equal(gpu.vertexBuffer, device.buffers[5]);
  assert.equal(gpu.vertexBuffer.usage, 128 | 32 | 4);
  assert.equal(gpu.vertexLayout.arrayStride, 40);
  assert.equal(gpu.vertexLayout.attributes.length, 3);
  assert.equal(gpu.vertexLayout.attributes[2].offset, 24);
  assert.equal(gpu.vertexLayout.attributes[2].format, "float32x4");
  assert.equal(gpu.worldMatrix[12], 5);
  assert.equal(gpu.version, 0);
  assert.equal(gpu.poseVersion, 0);
  gpu.dispose();
});

test("each pose is submitted before another pose overwrites palette and weights", async () => {
  const device = deviceSpy(),
    pose = scene(),
    gpu = await createGpuAnimationDeformer(device, pose, geometry());
  const buffer = gpu.vertexBuffer,
    world = gpu.worldMatrix;
  pose.sample(0.5);
  assert.equal(gpu.update(), gpu);
  pose.sample(1.5);
  gpu.update();
  await gpu.whenIdle();
  assert.deepEqual(
    device.submissions.map((s) => s.palette[12]),
    [-3, -2.5, -1.5],
  );
  assert.deepEqual(
    device.submissions.map((s) => s.weights[0]),
    [0, 0.25, 0.75],
  );
  assert.equal(gpu.version, 2);
  assert.equal(gpu.poseVersion, pose.version);
  assert.equal(gpu.vertexBuffer, buffer);
  assert.equal(gpu.worldMatrix, world);
  assert.equal(device.buffers.length, 7);
  assert.ok(
    device.writes.every((w) => w.buffer === device.buffers[3] || w.buffer === device.buffers[4]),
  );
  assert.deepEqual(
    device.writes.map((w) => w.bytes.length),
    [64, 4, 64, 4, 64, 4],
  );
  gpu.dispose();
});

test("shared skins upload only the selected node instance palette", async () => {
  const device = deviceSpy(),
    pose = scene();
  pose.sample(1);
  const gpu = await createGpuAnimationDeformer(device, pose, geometry(2));
  assert.equal(device.submissions[0].palette.length, 16);
  assert.equal(device.submissions[0].palette[12], 10);
  assert.equal(gpu.worldMatrix[12], -7);
  gpu.dispose();
});

test("static data is uploaded once and isolated from later input edits", async () => {
  const device = deviceSpy(),
    pose = scene(),
    source = geometry(),
    gpu = await createGpuAnimationDeformer(device, pose, source);
  source.positions[0] = 700;
  source.morphTargets[0].positions[0] = 900;
  source.weights[0] = 0;
  const original = device.buffers.slice(0, 3).map((b) => new Uint8Array(b.bytes).slice());
  for (let i = 0; i < 60; i++) {
    pose.sample(i / 30);
    gpu.update();
  }
  await gpu.whenIdle();
  assert.equal(device.buffers.length, 7);
  original.forEach((bytes, i) => assert.deepEqual(new Uint8Array(device.buffers[i].bytes), bytes));
  assert.equal(gpu.version, 60);
  gpu.dispose();
});

test("morph-only and plain geometry expose only present attributes and no skin uploads", async () => {
  for (const weights of [[], [-1, 2]]) {
    const pose = createAnimationPlayer({ format: "f3d-animation-v1", nodes: [{ weights }] });
    const device = deviceSpy(),
      source = {
        node: 0,
        positions: [1, 2, 3],
        morphTargets: weights.map(() => ({ positions: [1, 0, 0] })),
      };
    const gpu = await createGpuAnimationDeformer(device, pose, source);
    assert.equal(gpu.vertexLayout.attributes.length, 1);
    assert.equal(new Uint32Array(device.buffers[6].bytes)[2], 0);
    assert.equal(device.writes.length, weights.length ? 1 : 0);
    gpu.dispose();
  }
});

test("up to 32 influences are packed, including non-first joint indices", async () => {
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [{}, ...Array.from({ length: 32 }, () => ({}))],
    skins: [{ joints: Array.from({ length: 32 }, (_, i) => i + 1) }],
    instances: [{ node: 0, skin: 0 }],
  });
  const device = deviceSpy(),
    gpu = await createGpuAnimationDeformer(device, pose, {
      node: 0,
      positions: [1, 2, 3],
      joints: Array.from({ length: 32 }, (_, i) => i),
      weights: Array(32).fill(1 / 32),
      influences: 32,
    });
  const joints = new Uint32Array(device.buffers[2].bytes),
    weights = new Float32Array(device.buffers[2].bytes);
  for (let i = 0; i < 32; i++) {
    assert.equal(joints[i * 2], i);
    assert.equal(weights[i * 2 + 1], 1 / 32);
  }
  assert.equal(new Uint32Array(device.buffers[6].bytes)[2], 32);
  gpu.dispose();
});

test("dispatch spans two dimensions and pads only the last workgroup", async () => {
  const pose = createAnimationPlayer({ format: "f3d-animation-v1", nodes: [{}] }),
    device = deviceSpy();
  device.limits.maxComputeWorkgroupsPerDimension = 2;
  const gpu = await createGpuAnimationDeformer(device, pose, {
    node: 0,
    positions: Array(129 * 3).fill(1),
  });
  assert.deepEqual(device.submissions[0].dispatch, [2, 2]);
  assert.equal(new Uint32Array(device.buffers[6].bytes)[0], 129);
  gpu.dispose();
  const other = deviceSpy();
  other.limits.maxComputeWorkgroupsPerDimension = 2;
  await assert.rejects(
    createGpuAnimationDeformer(other, pose, { node: 0, positions: Array(257 * 3).fill(1) }),
    code("ANIMATION_GPU_LIMIT"),
  );
  assert.equal(other.buffers.length, 0);
});

for (const [name, value] of [
  ["maxStorageBuffersPerShaderStage", 5],
  ["maxBindingsPerBindGroup", 6],
  ["maxComputeWorkgroupSizeX", 32],
  ["maxComputeInvocationsPerWorkgroup", 32],
  ["maxUniformBuffersPerShaderStage", 0],
  ["maxStorageBufferBindingSize", 32],
  ["maxBufferSize", 32],
]) {
  test(`insufficient ${name} rejects before any GPU allocation`, async () => {
    const device = deviceSpy();
    device.limits[name] = value;
    await assert.rejects(
      createGpuAnimationDeformer(device, scene(), geometry()),
      code("ANIMATION_GPU_LIMIT"),
    );
    assert.equal(device.buffers.length, 0);
  });
}

test("exact device-buffer byte budget includes output and padding buffers", async () => {
  const device = deviceSpy(),
    gpu = await createGpuAnimationDeformer(device, scene(), geometry());
  const required = device.buffers.reduce((sum, b) => sum + b.size, 0);
  gpu.dispose();
  const limited = deviceSpy();
  await assert.rejects(
    createGpuAnimationDeformer(limited, scene(), geometry(), { maxBytes: required - 1 }),
    code("ANIMATION_GPU_LIMIT"),
  );
  assert.equal(limited.buffers.length, 0);
  const exact = await createGpuAnimationDeformer(deviceSpy(), scene(), geometry(), {
    maxBytes: required,
  });
  exact.dispose();
});

test("shape, joint, weight and f32 admission failures allocate no buffers", async () => {
  for (const patch of [
    { joints: [1, 0, 0, 0] },
    { weights: [0, 0, 0, 0] },
    { morphTargets: [] },
    { positions: [1e100, 0, 0] },
    { morphTargets: [{ colors: [1, 2, 3] }] },
  ]) {
    const device = deviceSpy();
    await assert.rejects(createGpuAnimationDeformer(device, scene(), { ...geometry(), ...patch }));
    assert.equal(device.buffers.length, 0);
  }
});

test("invalid dynamic values leave all queue inputs and submission stamps unchanged", async () => {
  const device = deviceSpy(),
    pose = scene(),
    gpu = await createGpuAnimationDeformer(device, pose, geometry());
  const beforeWrites = device.writes.length,
    beforeSubmits = device.submissions.length,
    world = [...gpu.worldMatrix];
  for (const corrupt of [
    () => {
      pose.morphWeights[0] = 1e100;
    },
    () => {
      pose.jointMatrices[0] = Infinity;
    },
    () => {
      pose.jointMatrices[3] = 1;
    },
    () => {
      pose.jointMatrices[12] = 1e38;
    },
    () => {
      pose.worldMatrices[0] = NaN;
    },
  ]) {
    pose.sample(1);
    corrupt();
    assert.throws(() => gpu.update(), code("ANIMATION_GPU_VALUE"));
    assert.equal(gpu.version, 0);
    assert.equal(gpu.poseVersion, 0);
    assert.deepEqual([...gpu.worldMatrix], world);
    assert.equal(device.writes.length, beforeWrites);
    assert.equal(device.submissions.length, beforeSubmits);
  }
  pose.sample(1);
  gpu.update();
  await gpu.whenIdle();
  assert.equal(gpu.version, 1);
  gpu.dispose();
});

test("detached pose storage and output matrix fail before queue writes", async () => {
  for (const target of ["jointMatrices", "morphWeights", "worldMatrices", "output"]) {
    const device = deviceSpy(),
      pose = scene(),
      gpu = await createGpuAnimationDeformer(device, pose, geometry());
    const storage = target === "output" ? gpu.worldMatrix : pose[target];
    structuredClone(storage.buffer, { transfer: [storage.buffer] });
    assert.throws(() => gpu.update(), code("ANIMATION_GPU_STORAGE"));
    assert.equal(device.submissions.length, 1);
    gpu.dispose();
  }
});

test("pipeline failure releases every owned buffer and leaves the pose live", async () => {
  const device = deviceSpy(),
    pose = scene();
  device.pipelineGate = deferred();
  const creating = createGpuAnimationDeformer(device, pose, geometry());
  assert.equal(device.scopes.length, 0);
  device.pipelineGate.reject(new Error("shader compile failed"));
  await assert.rejects(creating, /shader compile failed/);
  assert.equal(device.buffers.length, 7);
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  assert.equal(pose.disposed, false);
});

test("allocation validation errors fail initialization and release all buffers", async () => {
  const device = deviceSpy();
  device.scopeError = { message: "allocation refused" };
  await assert.rejects(
    createGpuAnimationDeformer(device, scene(), geometry()),
    code("ANIMATION_GPU_DEVICE"),
  );
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  assert.equal(device.scopes.length, 0);
});

test("asynchronous update validation errors are surfaced by whenIdle and latched", async () => {
  const device = deviceSpy(),
    gpu = await createGpuAnimationDeformer(device, scene(), geometry());
  device.scopeError = { message: "dispatch refused" };
  gpu.update();
  await assert.rejects(gpu.whenIdle(), code("ANIMATION_GPU_DEVICE"));
  assert.equal(gpu.failed, true);
  assert.throws(() => gpu.update(), code("ANIMATION_GPU_DEVICE"));
  gpu.dispose();
});

test("synchronous encoder failure does not publish new submission state", async () => {
  const device = deviceSpy(),
    pose = scene(),
    gpu = await createGpuAnimationDeformer(device, pose, geometry());
  pose.sample(1);
  device.encoderError = new Error("encoder failed");
  assert.throws(() => gpu.update(), /encoder failed/);
  assert.equal(gpu.version, 0);
  assert.equal(gpu.poseVersion, 0);
  assert.equal(device.submissions.length, 1);
  assert.equal(device.scopes.length, 0);
  gpu.dispose();
});

test("device loss during initialization races a still-pending pipeline and releases resources", async () => {
  const device = deviceSpy();
  device.pipelineGate = deferred();
  const creating = createGpuAnimationDeformer(device, scene(), geometry());
  device.loss.resolve({ message: "device removed" });
  await assert.rejects(creating, code("ANIMATION_GPU_LOST"));
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  device.pipelineGate.resolve();
});

test("device loss during completion cannot masquerade as a successful update", async () => {
  const device = deviceSpy(),
    gpu = await createGpuAnimationDeformer(device, scene(), geometry());
  device.completionGate = deferred();
  gpu.update();
  const pending = gpu.whenIdle();
  device.loss.resolve({ message: "lost while executing" });
  await assert.rejects(pending, code("ANIMATION_GPU_LOST"));
  assert.equal(gpu.failed, true);
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  gpu.dispose();
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  device.completionGate.resolve();
});

test("disposal is idempotent and owns neither the pose nor device", async () => {
  const device = deviceSpy(),
    pose = scene(),
    gpu = await createGpuAnimationDeformer(device, pose, geometry());
  gpu.dispose();
  gpu.dispose();
  assert.equal(gpu.disposed, true);
  assert.ok(device.buffers.every((b) => b.destroyed === 1));
  assert.equal(pose.disposed, false);
  pose.sample(1);
  assert.throws(() => gpu.update(), code("ANIMATION_GPU_DISPOSED"));
  await assert.rejects(gpu.whenIdle(), code("ANIMATION_GPU_DISPOSED"));
  const next = await createGpuAnimationDeformer(device, pose, geometry());
  pose.dispose();
  assert.throws(() => next.update(), code("ANIMATION_DISPOSED"));
  next.dispose();
});

test("compute shader does not use the reserved WGSL target identifier", () => {
  // WGSL section 16.2 reserves target even though JavaScript accepts it.
  // This catches the concrete compile blocker; it is not shader validation.
  assert.doesNotMatch(ANIMATION_DEFORM_WGSL, /\btarget\b/);
  assert.match(ANIMATION_DEFORM_WGSL, /var morph_index = 0u/);
});

test("whenIdle waits for earlier submission error scopes even if later work completes", async () => {
  const device = deviceSpy(),
    gpu = await createGpuAnimationDeformer(device, scene(), geometry());
  const earlier = deferred(),
    pop = device.popErrorScope.bind(device);
  let hold = true;
  device.popErrorScope = () => {
    const result = pop();
    if (hold) {
      hold = false;
      return earlier.promise;
    }
    return result;
  };
  gpu.update();
  gpu.update();
  let settled = false;
  const result = gpu.whenIdle().then(
    () => {
      settled = true;
      return null;
    },
    (error) => {
      settled = true;
      return error;
    },
  );
  await new Promise((resolve) => setImmediate(resolve));
  const prematurelySettled = settled;
  earlier.resolve({ message: "earlier dispatch was invalid" });
  const error = await result;
  gpu.dispose();
  assert.equal(
    prematurelySettled,
    false,
    "Later completion hid an outstanding earlier error scope",
  );
  assert.equal(error?.code, "ANIMATION_GPU_DEVICE");
});
