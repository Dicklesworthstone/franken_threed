import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildAnimation } from "./build_animation.mjs";
import { animationFixture, glbFixture } from "./fixtures/animation/gltf_fixture.mjs";

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
function fixture(kind = "gltf") {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-gpu-package-")),
    input = animationFixture();
  const entry = path.join(dir, "actor." + kind),
    out = path.join(dir, "built");
  if (kind === "glb") fs.writeFileSync(entry, glbFixture(input.model, input.bytes));
  else {
    fs.writeFileSync(entry, JSON.stringify(input.model));
    fs.writeFileSync(path.join(dir, "clip data.bin"), input.bytes);
  }
  return { ...input, dir, entry, out };
}
const cli = (...args) =>
  spawnSync(process.execPath, [fileURLToPath(new URL("./cli.mjs", import.meta.url)), ...args], {
    encoding: "utf8",
  });

for (const kind of ["gltf", "glb"])
  test(`${kind} GPU package relocates and imports without a GPU or original resources`, async () => {
    const f = fixture(kind),
      before = fs.readFileSync(f.entry),
      built = buildAnimation(f.entry, f.out, { webgpu: true });
    assert.equal(built.gpuEntry, "gpu_playback.mjs");
    assert.equal(built.gpuExecution, "webgpu-compute-f32");
    assert.equal(built.execution, "javascript-cpu-pose");
    assert.equal(built.accelerationClaim, false);
    assert.deepEqual(fs.readFileSync(f.entry), before);
    for (const artifact of built.artifacts) {
      const bytes = fs.readFileSync(path.join(f.out, artifact.file));
      assert.equal(bytes.length, artifact.bytes);
      assert.equal(hash(bytes), artifact.sha256);
    }
    assert.deepEqual(
      fs.readFileSync(path.join(f.out, "animation_webgpu.mjs")),
      fs.readFileSync(new URL("./animation_webgpu.mjs", import.meta.url)),
    );
    const moved = path.join(f.dir, "relocated");
    fs.renameSync(f.out, moved);
    fs.renameSync(f.entry, f.entry + ".unavailable");
    if (kind === "gltf")
      fs.renameSync(path.join(f.dir, "clip data.bin"), path.join(f.dir, "clip data.unavailable"));
    const api = await import(pathToFileURL(path.join(moved, built.gpuEntry)));
    const cpu = await import(pathToFileURL(path.join(moved, built.playbackEntry)));
    assert.equal(api.createPlayer, cpu.createPlayer);
    assert.equal(api.createAnimationController, cpu.createAnimationController);
    assert.equal(api.createAnimationDeformer, cpu.createAnimationDeformer);
    assert.equal(typeof api.createGpuAnimationDeformer, "function");
    assert.equal(typeof api.createGpuBufferGeometry, "function");
    assert.deepEqual(
      fs.readFileSync(path.join(moved, "gpu_buffer_geometry.mjs")),
      fs.readFileSync(new URL("./gpu_buffer_geometry.mjs", import.meta.url)),
    );
    assert.throws(() => api.createGpuBufferGeometry(undefined, {}), {
      code: "GEOMETRY_GPU_DEVICE",
    });
    const pose = api.createPlayer(),
      control = api.createAnimationController(pose);
    control.createAction(0).play();
    control.update(1);
    assert.equal(pose.jointMatrices[12], -4);
    assert.equal(pose.jointMatrices[13], 1);
    assert.equal(pose.morphWeights[0], 0.5);
    // No device request or automatic fallback occurs at import/build time.
    await assert.rejects(api.createGpuAnimationDeformer(undefined, pose, {}), {
      code: "ANIMATION_GPU_DEVICE",
    });
    assert.equal(pose.disposed, false);
    control.dispose();
    pose.dispose();
  });

test("default CPU builds retain their exact file list and no GPU dependency edges", () => {
  const f = fixture(),
    built = buildAnimation(f.entry, f.out);
  assert.equal(built.gpuEntry, undefined);
  assert.equal(built.gpuExecution, undefined);
  assert.deepEqual(fs.readdirSync(f.out).sort(), [
    "animation.json",
    "animation.mjs",
    "animation_controller.mjs",
    "animation_deformer.mjs",
    "animation_runtime.mjs",
    "manifest.json",
    "playback.mjs",
  ]);
  for (const file of ["animation.mjs", "playback.mjs"])
    assert.doesNotMatch(fs.readFileSync(path.join(f.out, file), "utf8"), /webgpu|gpu_playback/);
  const gpu = fixture();
  buildAnimation(gpu.entry, gpu.out, { webgpu: true });
  for (const file of built.emittedFiles.filter((file) => file !== "manifest.json")) {
    assert.deepEqual(
      fs.readFileSync(path.join(f.out, file)),
      fs.readFileSync(path.join(gpu.out, file)),
    );
  }
});

test("GPU artifact and entry sizes are included before output publication", () => {
  const f = fixture(),
    built = buildAnimation(f.entry, f.out, { webgpu: true }),
    next = fixture();
  assert.throws(
    () => buildAnimation(next.entry, next.out, { webgpu: true, maxBytes: built.outputBytes - 1 }),
    { code: "GLTF_ANIMATION_LIMIT" },
  );
  assert.equal(fs.existsSync(next.out), false);
  const exact = fixture();
  assert.equal(
    buildAnimation(exact.entry, exact.out, { webgpu: true, maxBytes: built.outputBytes })
      .outputBytes,
    built.outputBytes,
  );
  for (const file of ["animation_webgpu.mjs", "gpu_playback.mjs"])
    assert.ok(built.artifacts.find((a) => a.file === file));
});

test("GPU packages are deterministic across source/output roots", () => {
  const a = fixture(),
    b = fixture();
  const built = buildAnimation(a.entry, a.out, { webgpu: true });
  buildAnimation(b.entry, b.out, { webgpu: true });
  for (const file of built.emittedFiles)
    assert.deepEqual(
      fs.readFileSync(path.join(a.out, file)),
      fs.readFileSync(path.join(b.out, file)),
    );
});

test("invalid GPU build options fail before creating output", () => {
  for (const webgpu of [null, 1, "true", {}]) {
    const f = fixture();
    assert.throws(() => buildAnimation(f.entry, f.out, { webgpu }), /webgpu must be boolean/);
    assert.equal(fs.existsSync(f.out), false);
  }
});

for (const kind of ["gltf", "glb"])
  test(`CLI builds ${kind} with the GPU entry only when explicitly requested`, () => {
    const f = fixture(kind),
      result = cli("--entry", f.entry, "--build-animation", f.out, "--animation-webgpu");
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /gpu_playback\.mjs/);
    assert.match(result.stdout, /no speedup claim/);
    const manifest = JSON.parse(fs.readFileSync(path.join(f.out, "manifest.json"), "utf8"));
    assert.equal(manifest.gpuEntry, "gpu_playback.mjs");
    assert.equal(fs.existsSync(path.join(f.out, manifest.gpuEntry)), true);
    const other = fixture(kind),
      plain = cli("--entry", other.entry, "--build-animation", other.out);
    assert.equal(plain.status, 0, plain.stderr);
    assert.equal(fs.existsSync(path.join(other.out, "animation_webgpu.mjs")), false);
  });

test("CLI refuses GPU flag without animation build before importing unrelated tooling", () => {
  for (const extra of [
    [],
    ["--build-app", "app"],
    ["--build-kernel", "kernel", "--parameter-types", "f64[]"],
  ]) {
    const f = fixture(),
      result = cli("--entry", f.entry, "--animation-webgpu", ...extra);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /--animation-webgpu requires --build-animation/);
    assert.equal(fs.existsSync(f.out), false);
  }
});

test("CLI preserves incompatible build and destination safety rules", () => {
  const f = fixture();
  for (const extra of [
    ["--output", path.join(f.dir, "report.json")],
    ["--build-app", path.join(f.dir, "app")],
  ]) {
    const result = cli(
      "--entry",
      f.entry,
      "--build-animation",
      f.out,
      "--animation-webgpu",
      ...extra,
    );
    assert.equal(result.status, 1);
    assert.match(result.stderr, /cannot be combined/);
    assert.equal(fs.existsSync(f.out), false);
  }
  fs.mkdirSync(f.out);
  fs.writeFileSync(path.join(f.out, "keep"), "existing data");
  const result = cli("--entry", f.entry, "--build-animation", f.out, "--animation-webgpu");
  assert.equal(result.status, 1);
  assert.match(result.stderr, /ANIMATION_OUTPUT_EXISTS/);
  assert.deepEqual(fs.readdirSync(f.out), ["keep"]);
  assert.equal(fs.readFileSync(path.join(f.out, "keep"), "utf8"), "existing data");
});

test("CLI help advertises the opt-in animation GPU flag", () => {
  const result = cli("--help");
  assert.equal(result.status, 0);
  assert.match(result.stdout, /--animation-webgpu/);
});
