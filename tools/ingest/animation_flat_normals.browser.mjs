/** Native WebGPU coverage for the second deformation pass. Called by the
 * existing animation_webgpu E2E page; a missing device is never a test pass.
 * CPU/GPU comparisons require actual readback, not a command spy or timing claim.
 */

import { createAnimationDeformer } from "./animation_deformer.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { createGpuAnimationDeformer } from "./animation_webgpu.mjs";

const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
export async function runFlatNormalWebGpuChecks(device) {
  check(device?.queue, "A real WebGPU device is required");
  const resources = new Set(),
    errors = [],
    results = [];
  const report = (event) => errors.push(event.error.message);
  device.addEventListener("uncapturederror", report);
  let pose, cpu, gpu;
  function enqueueRead() {
    const size = gpu.vertexCount * 40;
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    resources.add(buffer);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(gpu.vertexBuffer, 0, buffer, 0, size);
    device.queue.submit([encoder.finish()]);
    return buffer;
  }
  function expected() {
    const values = new Float32Array(cpu.vertexCount * 10);
    for (let v = 0; v < cpu.vertexCount; v++) {
      values.set(cpu.positions.subarray(v * 3, v * 3 + 3), v * 10);
      values.set(cpu.normals.subarray(v * 3, v * 3 + 3), v * 10 + 3);
      values[v * 10 + 9] = 1;
    }
    return values;
  }
  async function compare(buffer, reference) {
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(buffer.getMappedRange());
      check(values.length === reference.length, "Flat normal readback extent differs");
      for (let i = 0; i < values.length; i++)
        check(
          Number.isFinite(values[i]) &&
            Math.abs(values[i] - reference[i]) <= 2e-5 * Math.max(1, Math.abs(reference[i])),
          `Flat normal GPU mismatch at ${i}: ${values[i]} versus ${reference[i]}`,
        );
    } finally {
      buffer.unmap();
      buffer.destroy();
      resources.delete(buffer);
    }
  }
  try {
    for (const scale of [1e-20, 1, 1e20]) {
      pose = createAnimationPlayer({
        format: "f3d-animation-v1",
        nodes: [{ weights: [0] }, {}, {}],
        skins: [{ joints: [1, 2] }],
        instances: [{ node: 0, skin: 0 }],
        clips: [
          {
            channels: [
              { node: 0, path: "weights", times: [0, 1], values: [0, 1] },
              { node: 2, path: "translation", times: [0, 1], values: [0, 0, 0, 0, 0, 2 * scale] },
            ],
          },
        ],
      });
      const source = {
        node: 0,
        flatNormals: true,
        positions: [],
        morphTargets: [{ positions: [] }],
        joints: [],
        weights: [],
        influences: 1,
      };
      // 195 vertices cross deformation workgroups; 65 triangles cross normal
      // workgroups. Alternate winding and retain collapsed triangles.
      for (let t = 0; t < 65; t++)
        for (const v of t % 2 ? [0, 2, 1] : [0, 1, 2]) {
          const collapsed = t % 7 === 0;
          source.positions.push(v === 1 ? scale : 0, v === 2 && !collapsed ? scale : 0, 0);
          source.morphTargets[0].positions.push(0, 0, v === 2 && !collapsed ? scale : 0);
          source.joints.push(v === 1 ? 1 : 0);
          source.weights.push(1);
        }
      cpu = createAnimationDeformer(pose, source);
      gpu = await createGpuAnimationDeformer(device, pose, source);
      const originalBuffer = gpu.vertexBuffer;
      for (const time of [0, 0.5, 1]) {
        pose.sample(time);
        cpu.update();
        gpu.update();
        await compare(enqueueRead(), expected());
        await gpu.whenIdle();
        check(gpu.vertexBuffer === originalBuffer, "Dynamic normals replaced the vertex buffer");
      }
      // Preserve both observations on the queue without waiting between poses.
      pose.sample(0.25);
      cpu.update();
      gpu.update();
      const a = enqueueRead(),
        ra = expected();
      pose.sample(0.75);
      cpu.update();
      gpu.update();
      const b = enqueueRead(),
        rb = expected();
      await compare(a, ra);
      await compare(b, rb);
      await gpu.whenIdle();
      results.push({ scale, triangles: 65, poses: 5, sameFrameVersions: true });
      gpu.dispose();
      cpu.dispose();
      pose.dispose();
      gpu = cpu = pose = null;
    }
    await device.queue.onSubmittedWorkDone();
    check(errors.length === 0, errors.join("\n"));
    return {
      status: "passed",
      cases: results,
      execution: "actual WebGPU deformation and flat-normal readback",
      speedupClaim: false,
    };
  } finally {
    device.removeEventListener("uncapturederror", report);
    gpu?.dispose();
    cpu?.dispose();
    pose?.dispose();
    for (const buffer of resources) buffer.destroy();
  }
}
