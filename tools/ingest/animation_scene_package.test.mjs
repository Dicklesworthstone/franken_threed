import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { buildAnimation } from "./build_animation.mjs";
import { animationFixture, glbFixture } from "./fixtures/animation/gltf_fixture.mjs";

const hash = (b) => createHash("sha256").update(b).digest("hex");
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function input(kind = "gltf") {
  const f = animationFixture(),
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-render-package-"));
  const entry = path.join(dir, "actor." + kind),
    out = path.join(dir, "out");
  if (kind === "glb") fs.writeFileSync(entry, glbFixture(f.model, f.bytes));
  else {
    fs.writeFileSync(entry, JSON.stringify(f.model));
    fs.writeFileSync(path.join(dir, "clip data.bin"), f.bytes);
  }
  return { ...f, dir, entry, out };
}
// Counts host submissions, without pretending to execute compute or rasterize.
function device() {
  const buffers = [],
    submitted = [],
    scopes = [];
  return {
    buffers,
    submitted,
    lost: new Promise(() => {}),
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 1 << 27,
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
    pushErrorScope: (filter) => scopes.push(filter),
    popErrorScope() {
      assert.ok(scopes.pop());
      return Promise.resolve(null);
    },
    createBuffer({ size }) {
      const b = {
        data: new ArrayBuffer(size),
        size,
        destroyed: false,
        getMappedRange() {
          return this.data;
        },
        unmap() {},
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(b);
      return b;
    },
    createShaderModule: (v) => v,
    createBindGroupLayout: (v) => v,
    createPipelineLayout: (v) => v,
    createBindGroup: (v) => v,
    createComputePipelineAsync: async (v) => ({ ...v, getBindGroupLayout: () => ({}) }),
    createRenderPipelineAsync: async (v) => v,
    createCommandEncoder() {
      let kind;
      const pass = {
        setPipeline() {},
        setBindGroup() {},
        setVertexBuffer() {},
        setIndexBuffer() {},
        dispatchWorkgroups() {},
        draw() {},
        drawIndexed() {},
        end() {},
      };
      return {
        beginComputePass() {
          kind = "compute";
          return pass;
        },
        beginRenderPass() {
          kind = "render";
          return pass;
        },
        finish() {
          return kind;
        },
      };
    },
    queue: {
      writeBuffer() {},
      submit(commands) {
        submitted.push(...commands);
      },
      onSubmittedWorkDone: async () => {},
    },
  };
}
for (const kind of ["gltf", "glb"])
  test(`${kind} emitted scene actually initializes, advances and submits after relocation`, async () => {
    const f = input(kind),
      built = buildAnimation(f.entry, f.out, { webgpu: true });
    const primitive = f.model.meshes[0].primitives[0];
    function attr(index) {
      const a = f.model.accessors[index],
        v = f.model.bufferViews[a.bufferView],
        C = a.componentType === 5121 ? Uint8Array : Float32Array;
      return new C(
        f.bytes.buffer,
        f.bytes.byteOffset + (v.byteOffset ?? 0) + (a.byteOffset ?? 0),
        a.count * (a.type === "VEC3" ? 3 : 4),
      ).slice();
    }
    const drawable = {
      geometry: {
        node: 0,
        positions: attr(primitive.attributes.POSITION),
        joints: attr(primitive.attributes.JOINTS_0),
        weights: attr(primitive.attributes.WEIGHTS_0),
        morphTargets: primitive.targets.map((t) => ({ positions: attr(t.POSITION) })),
      },
      indices: [0, 1, 2],
      baseColor: [1, 0, 0, 1],
    };
    const moved = path.join(f.dir, "deployed");
    fs.renameSync(f.out, moved);
    fs.renameSync(f.entry, f.entry + ".unavailable");
    if (kind === "gltf")
      fs.renameSync(path.join(f.dir, "clip data.bin"), path.join(f.dir, "clip data.unavailable"));
    const api = await import(pathToFileURL(path.join(moved, built.gpuEntry)));
    assert.equal(typeof api.createGpuAnimationRenderer, "function");
    assert.equal(typeof api.createGpuAnimationScene, "function");
    const p = api.createPlayer(),
      d = device(),
      scene = await api.createGpuAnimationScene(d, p, [drawable]);
    const original = scene.deformers[0].vertexBuffer,
      allocationCount = d.buffers.length;
    const action = scene.controller
      .createAction(0, { loop: "once", clampWhenFinished: true })
      .play();
    d.submitted.length = 0;
    for (let i = 0; i < 60; i++) {
      scene.update(1 / 30);
      scene.render({ colorView: {}, depthView: {}, viewProjection: identity() });
    }
    await scene.whenIdle();
    assert.equal(action.finished, true);
    assert.equal(scene.poseVersion, p.version);
    assert.equal(d.buffers.length, allocationCount);
    assert.equal(scene.deformers[0].vertexBuffer, original);
    assert.equal(d.submitted.length, 120);
    for (let i = 0; i < 60; i++)
      assert.deepEqual(d.submitted.slice(i * 2, i * 2 + 2), ["compute", "render"]);
    assert.equal(p.jointMatrices[12], -3);
    assert.equal(p.jointMatrices[13], 2);
    for (const artifact of built.artifacts) {
      const bytes = fs.readFileSync(path.join(moved, artifact.file));
      assert.equal(bytes.length, artifact.bytes);
      assert.equal(hash(bytes), artifact.sha256);
    }
    scene.dispose();
    assert.ok(d.buffers.every((b) => b.destroyed));
    assert.equal(p.disposed, false);
    p.dispose();
  });

test("render/scene exports are opt-in and use exactly the package-owned implementations", async () => {
  const plain = input(),
    cpu = buildAnimation(plain.entry, plain.out),
    f = input(),
    gpu = buildAnimation(f.entry, f.out, { webgpu: true });
  for (const name of ["animation_webgpu.mjs", "animation_render.mjs", "animation_scene.mjs"]) {
    assert.equal(fs.existsSync(path.join(plain.out, name)), false);
    assert.deepEqual(
      fs.readFileSync(path.join(f.out, name)),
      fs.readFileSync(new URL("./" + name, import.meta.url)),
    );
  }
  for (const file of cpu.emittedFiles.filter((f) => f !== "manifest.json"))
    assert.deepEqual(
      fs.readFileSync(path.join(plain.out, file)),
      fs.readFileSync(path.join(f.out, file)),
    );
  const api = await import(pathToFileURL(path.join(f.out, gpu.gpuEntry)));
  const direct = await import(pathToFileURL(path.join(f.out, "animation_scene.mjs")));
  assert.equal(api.createGpuAnimationScene, direct.createGpuAnimationScene);
  assert.equal(cpu.gpuRendering, undefined);
  assert.match(gpu.gpuRendering, /explicit-unlit/);
  assert.equal(gpu.accelerationClaim, false);
});

test("new GPU modules count toward deterministic pre-write output limits", () => {
  const a = input(),
    b = input(),
    x = buildAnimation(a.entry, a.out, { webgpu: true }),
    y = buildAnimation(b.entry, b.out, { webgpu: true });
  assert.deepEqual(x.artifacts, y.artifacts);
  assert.equal(x.outputBytes, y.outputBytes);
  const small = input();
  assert.throws(
    () => buildAnimation(small.entry, small.out, { webgpu: true, maxBytes: x.outputBytes - 1 }),
    { code: "GLTF_ANIMATION_LIMIT" },
  );
  assert.equal(fs.existsSync(small.out), false);
  const exact = input();
  assert.equal(
    buildAnimation(exact.entry, exact.out, { webgpu: true, maxBytes: x.outputBytes }).outputBytes,
    x.outputBytes,
  );
});
