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
function files(kind = "gltf", mutate = () => {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-animation-build-")),
    f = animationFixture();
  mutate(f);
  const entry = path.join(dir, "actor." + kind),
    out = path.join(dir, "player");
  if (kind === "glb") fs.writeFileSync(entry, glbFixture(f.model, f.bytes));
  else {
    fs.writeFileSync(entry, JSON.stringify(f.model));
    fs.writeFileSync(path.join(dir, "clip data.bin"), f.bytes);
  }
  return { ...f, dir, entry, out };
}
for (const kind of ["gltf", "glb"])
  test(`${kind} creates a relocatable player from actual model resources`, async () => {
    const f = files(kind),
      source = fs.readFileSync(f.entry),
      result = buildAnimation(f.entry, f.out);
    assert.equal(result.nodeCount, 3);
    assert.equal(result.clips.length, 2);
    assert.equal(result.instances[0].jointCount, 2);
    assert.equal(result.source.sha256, hash(source));
    assert.deepEqual(fs.readFileSync(f.entry), source);
    for (const a of result.artifacts)
      assert.equal(hash(fs.readFileSync(path.join(f.out, a.file))), a.sha256);
    const relocated = path.join(f.dir, "moved");
    fs.renameSync(f.out, relocated);
    fs.renameSync(f.entry, f.entry + ".unavailable");
    const { createPlayer } = await import(pathToFileURL(path.join(relocated, "animation.mjs")));
    const a = createPlayer(),
      b = createPlayer();
    a.sample(1);
    assert.equal(a.jointMatrices[12], -4);
    assert.equal(a.jointMatrices[13], 1);
    assert.equal(b.jointMatrices[12], -5);
    assert.equal(a.morphWeights[0], 0.5);
    assert.equal(a.morphWeights[1], 0.5);
    assert.deepEqual(fs.readdirSync(relocated).sort(), [
      "animation.json",
      "animation.mjs",
      "animation_controller.mjs",
      "animation_deformer.mjs",
      "animation_runtime.mjs",
      "manifest.json",
      "playback.mjs",
    ]);
    assert.ok(!fs.readFileSync(path.join(relocated, "animation.mjs"), "utf8").includes(f.dir));
  });

for (const form of ["base64", "escaped"])
  test(`embedded ${form} buffer URI is decoded without network`, async () => {
    const f = files("gltf", (f) => {
      f.model.buffers[0].uri =
        form === "base64"
          ? "data:application/octet-stream;base64," + Buffer.from(f.bytes).toString("base64")
          : "data:application/gltf-buffer," +
            [...f.bytes].map((b) => "%" + b.toString(16).padStart(2, "0")).join("");
    });
    const result = buildAnimation(f.entry, f.out);
    assert.equal(result.dependencies[0].uri, "#data-buffer-0");
    const { createPlayer } = await import(pathToFileURL(path.join(f.out, "animation.mjs")));
    assert.equal(createPlayer().sample(1).jointMatrices[12], -4);
  });

test("percent encoded paths, JSON names and signed zero survive package generation", async () => {
  const f = files("gltf", (f) => {
    f.model.buffers[0].uri = "clip%20data.bin?data=1#buffer";
    f.model.animations[0].name = '";globalThis.bad=true;//</script>';
  });
  const text = fs
    .readFileSync(f.entry, "utf8")
    .replace('"translation":[5,0,0]', '"translation":[5,-0,0]');
  fs.writeFileSync(f.entry, text);
  const result = buildAnimation(f.entry, f.out);
  const { createPlayer } = await import(pathToFileURL(path.join(f.out, "animation.mjs")));
  const p = createPlayer();
  assert.ok(Object.is(p.translations[1], -0));
  assert.equal(p.clips[0].name, f.model.animations[0].name);
  assert.equal(globalThis.bad, undefined);
  assert.equal(result.dependencies[0].uri, "clip data.bin?data=1#buffer");
});

test("unsupported channels, sparse bounds and singular bind poses leave no output", () => {
  for (const change of [
    (f) => {
      f.model.animations[0].channels[0].target.extensions = { KHR_animation_pointer: {} };
    },
    (f) => {
      f.model.nodes[0].scale = [0, 0, 0];
    },
    (f) => {
      f.model.bufferViews[0].byteLength = 1;
    },
  ]) {
    const f = files("gltf", change);
    assert.throws(() => buildAnimation(f.entry, f.out));
    assert.equal(fs.existsSync(f.out), false);
  }
});

test("preexisting directories, files and symlinks are never overwritten", () => {
  for (const type of ["directory", "file", "symlink"]) {
    const f = files();
    if (type === "directory") fs.mkdirSync(f.out);
    else if (type === "file") fs.writeFileSync(f.out, "KEEP");
    else fs.symlinkSync(f.entry, f.out);
    assert.throws(() => buildAnimation(f.entry, f.out), { code: "ANIMATION_OUTPUT_EXISTS" });
    if (type === "file") assert.equal(fs.readFileSync(f.out, "utf8"), "KEEP");
  }
});

for (const uri of [
  "https://example.invalid/clip.bin",
  "../clip.bin",
  "data:application/octet-stream;base64,A",
  "data:application/octet-stream,%GG",
])
  test(`unclosed or malformed buffer URI is rejected: ${uri}`, () => {
    const f = files("gltf", (f) => {
      f.model.buffers[0].uri = uri;
    });
    assert.throws(() => buildAnimation(f.entry, f.out));
    assert.equal(fs.existsSync(f.out), false);
  });

test("symlink escape and byte/component budgets are enforced before output", () => {
  const f = files(),
    outside = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-outside-"));
  fs.writeFileSync(path.join(outside, "data"), f.bytes);
  fs.renameSync(path.join(f.dir, "clip data.bin"), path.join(f.dir, "original.bin"));
  fs.symlinkSync(path.join(outside, "data"), path.join(f.dir, "clip data.bin"));
  assert.throws(() => buildAnimation(f.entry, f.out), { code: "GLTF_ANIMATION_ROOT" });
  const next = files();
  assert.throws(() => buildAnimation(next.entry, next.out, { maxBytes: 10 }), {
    code: "GLTF_ANIMATION_LIMIT",
  });
  assert.throws(() => buildAnimation(next.entry, next.out, { maxComponents: 2 }), {
    code: "GLTF_ANIMATION_LIMIT",
  });
  assert.equal(fs.existsSync(next.out), false);
});

test("malformed GLB header/chunks do not masquerade as a usable animation", () => {
  for (const change of [
    (b) => b.writeUInt32LE(1, 4),
    (b) => b.writeUInt32LE(10, 8),
    (b) => b.writeUInt32LE(0, 16),
    (b) => b.writeUInt32LE(3, 12),
  ]) {
    const f = files("glb"),
      b = fs.readFileSync(f.entry);
    change(b);
    fs.writeFileSync(f.entry, b);
    assert.throws(() => buildAnimation(f.entry, f.out), { code: "GLTF_ANIMATION_GLB" });
    assert.equal(fs.existsSync(f.out), false);
  }
});

test("package generation is deterministic across relocated source roots", () => {
  const a = files(),
    b = files();
  const ar = buildAnimation(a.entry, a.out),
    br = buildAnimation(b.entry, b.out);
  for (const file of ar.emittedFiles)
    assert.deepEqual(
      fs.readFileSync(path.join(a.out, file)),
      fs.readFileSync(path.join(b.out, file)),
    );
  assert.deepEqual(ar.artifacts, br.artifacts);
});

for (const kind of ["gltf", "glb"])
  test(`${kind} playback package drives deformed geometry after all source resources move`, async () => {
    const f = files(kind),
      result = buildAnimation(f.entry, f.out),
      moved = path.join(f.dir, "deployed");
    assert.equal(result.playbackEntry, "playback.mjs");
    assert.equal(result.accelerationClaim, false);
    // The model loader provides decoded geometry; the package must not reload it.
    const primitive = f.model.meshes[0].primitives[0];
    function attribute(index) {
      const a = f.model.accessors[index],
        v = f.model.bufferViews[a.bufferView];
      const C = a.componentType === 5121 ? Uint8Array : Float32Array;
      return new C(
        f.bytes.buffer,
        f.bytes.byteOffset + (v.byteOffset ?? 0) + (a.byteOffset ?? 0),
        a.count * { VEC3: 3, VEC4: 4 }[a.type],
      );
    }
    const geometry = {
      node: 0,
      positions: attribute(primitive.attributes.POSITION),
      joints: attribute(primitive.attributes.JOINTS_0),
      weights: attribute(primitive.attributes.WEIGHTS_0),
      morphTargets: primitive.targets.map((t) => ({ positions: attribute(t.POSITION) })),
    };
    fs.renameSync(f.out, moved);
    fs.renameSync(f.entry, f.entry + ".unavailable");
    if (kind === "gltf")
      fs.renameSync(path.join(f.dir, "clip data.bin"), path.join(f.dir, "clip data.unavailable"));
    const api = await import(pathToFileURL(path.join(moved, result.playbackEntry)));
    const sampler = await import(pathToFileURL(path.join(moved, result.entry)));
    assert.equal(api.createPlayer, sampler.createPlayer);
    const pose = api.createPlayer(),
      control = api.createAnimationController(pose),
      mesh = api.createAnimationDeformer(pose, geometry);
    const positions = mesh.positions,
      world = mesh.worldMatrix,
      bounds = mesh.bounds;
    const move = control.createAction(0, { loop: "once", clampWhenFinished: true }).play();
    for (let frame = 1; frame <= 60; frame++) {
      control.update(1 / 30);
      mesh.update();
      const t = Math.min(frame / 30, 2);
      for (let vertex = 0; vertex < 3; vertex++) {
        const i = vertex * 3;
        assert.ok(Math.abs(mesh.positions[i] - (geometry.positions[i] - 5 + t)) < 1e-5);
        assert.ok(Math.abs(mesh.positions[i + 1] - (geometry.positions[i + 1] + t)) < 1e-5);
        assert.equal(mesh.positions[i + 2], geometry.positions[i + 2]);
      }
      assert.equal(mesh.positions, positions);
      assert.equal(mesh.worldMatrix, world);
      assert.equal(mesh.bounds, bounds);
      assert.equal(mesh.poseVersion, pose.version);
    }
    assert.equal(move.finished, true);
    const rotate = control.createAction(1, { loop: "once", clampWhenFinished: true });
    control.crossFade(move, rotate, 1);
    control.update(0.5);
    mesh.update();
    assert.equal(move.weight, 0.5);
    assert.equal(rotate.weight, 0.5);
    // Independently blend the same clips through the sampler-only entry.
    const reference = sampler.createPlayer();
    reference.blend([
      { clip: 0, time: 2, weight: 0.5 },
      { clip: 1, time: 0.5, weight: 0.5 },
    ]);
    const expected = api.createAnimationDeformer(reference, geometry);
    assert.deepEqual(mesh.positions, expected.positions);
    assert.deepEqual(mesh.worldMatrix, expected.worldMatrix);
    control.update(0.5);
    mesh.update();
    assert.equal(move.playing, false);
    assert.equal(rotate.weight, 1);
    control.dispose();
    mesh.dispose();
    assert.equal(pose.disposed, false);
    pose.sample(0);
    assert.equal(fs.existsSync(f.entry), false);
  });

test("playback modules are hashed byte-for-byte and included in the output budget", () => {
  const f = files(),
    result = buildAnimation(f.entry, f.out);
  for (const name of ["animation_controller.mjs", "animation_deformer.mjs"]) {
    const expected = fs.readFileSync(new URL("./" + name, import.meta.url));
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), expected);
    const artifact = result.artifacts.find((a) => a.file === name);
    assert.equal(artifact.bytes, expected.length);
    assert.equal(artifact.sha256, hash(expected));
  }
  const next = files();
  assert.throws(() => buildAnimation(next.entry, next.out, { maxBytes: result.outputBytes - 1 }), {
    code: "GLTF_ANIMATION_LIMIT",
  });
  assert.equal(fs.existsSync(next.out), false);
  const exact = files();
  assert.equal(
    buildAnimation(exact.entry, exact.out, { maxBytes: result.outputBytes }).outputBytes,
    result.outputBytes,
  );
});

test("sampler-only module retains its single runtime import", () => {
  const f = files();
  buildAnimation(f.entry, f.out);
  const source = fs.readFileSync(path.join(f.out, "animation.mjs"), "utf8");
  assert.match(source, /from '\.\/animation_runtime\.mjs'/);
  assert.equal(source.includes("animation_controller"), false);
  assert.equal(source.includes("animation_deformer"), false);
});
