import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
// Actual builder, scene, ordering and bounds; explicit pose/decoder and GPU
// boundaries isolate emitted dependency closure, not binary import or rendering.
async function setup() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-culling-package-")),
    toolkit = path.join(root, "toolkit");
  fs.mkdirSync(toolkit);
  for (const name of [
    "build_animation.mjs",
    "animation_scene.mjs",
    "animation_draw_order.mjs",
    "animation_bounds.mjs",
    "animation_shadow.mjs",
    "animation_shadow_receiver.mjs",
    "animation_scene_shadow.mjs",
    "animation_shadow_view.mjs",
  ])
    fs.copyFileSync(new URL("./" + name, import.meta.url), path.join(toolkit, name));
  const runtime = `export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export function createAnimationPlayer(def){const p={nodeCount:def.nodes.length,instances:[],clips:[],version:0,disposed:false,
 morphOffsets:new Uint32Array(def.nodes.length+1),morphWeights:new Float64Array(),jointMatrices:new Float64Array(),
 worldMatrices:new Float64Array(def.nodes.flatMap(n=>[1,0,0,0,0,1,0,0,0,0,1,0,...(n.translation??[0,0,0]),1])),dispose(){this.disposed=true;}};return p;}`;
  fs.writeFileSync(path.join(toolkit, "animation_runtime.mjs"), runtime);
  fs.writeFileSync(
    path.join(toolkit, "animation_gltf.mjs"),
    "export function decodeGltfAnimation(model){return model;}",
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_controller.mjs"),
    "export function createAnimationController(pose){return {update(){pose.version++;},dispose(){}};}",
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_deformer.mjs"),
    'export function createAnimationDeformer(){throw Error("unused CPU boundary");}',
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_webgpu.mjs"),
    "export async function createGpuAnimationDeformer(d,p,g){return d.deformer(p,g);}",
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_render.mjs"),
    `export class AnimationRenderError extends Error{constructor(code,message){super(message);this.code=code;}}
export async function createGpuAnimationRenderer(d){return d.renderer;}`,
  );
  const entry = path.join(root, "model.gltf");
  fs.writeFileSync(
    entry,
    JSON.stringify({ asset: { version: "2.0" }, nodes: [{}, { translation: [20, 0, 0] }] }),
  );
  const { buildAnimation } = await import(pathToFileURL(path.join(toolkit, "build_animation.mjs")));
  return { root, toolkit, entry, buildAnimation };
}
function device() {
  const frames = [];
  let next = 0;
  return {
    frames,
    renderer: {
      allocatedBytes: 0,
      failed: false,
      async addMesh(g) {
        return { id: next++, dispose() {} };
      },
      render(f) {
        frames.push(f);
      },
      async whenIdle() {},
      dispose() {},
    },
    deformer(p, g) {
      return {
        bufferBytes: 120,
        poseVersion: p.version,
        worldMatrix: new Float64Array(p.worldMatrices.slice(g.node * 16, g.node * 16 + 16)),
        update() {
          this.poseVersion = p.version;
          this.worldMatrix.set(p.worldMatrices.slice(g.node * 16, g.node * 16 + 16));
        },
        async whenIdle() {},
        dispose() {},
      };
    },
  };
}
test("a relocated generated GPU package culls without original source modules", async () => {
  const f = await setup(),
    out = path.join(f.root, "package"),
    built = f.buildAnimation(f.entry, out, { webgpu: true });
  const record = built.artifacts.find((x) => x.file === "animation_bounds.mjs");
  assert.ok(record);
  assert.equal(
    record.sha256,
    hash(fs.readFileSync(new URL("./animation_bounds.mjs", import.meta.url))),
  );
  assert.deepEqual(
    fs.readFileSync(path.join(out, record.file)),
    fs.readFileSync(new URL("./" + record.file, import.meta.url)),
  );
  const moved = path.join(f.root, "deployed");
  fs.renameSync(out, moved);
  fs.renameSync(f.toolkit, f.toolkit + ".unavailable");
  fs.renameSync(f.entry, f.entry + ".unavailable");
  const api = await import(pathToFileURL(path.join(moved, built.gpuEntry))),
    p = api.createPlayer(),
    d = device();
  const items = [0, 1].map((node) => ({
    geometry: { node, positions: [0, 0, 0.5, 0.1, 0, 0.5, 0, 0.1, 0.5] },
  }));
  const s = await api.createGpuAnimationScene(d, p, items, { frustumCulling: true });
  s.render({ colorView: {}, depthView: {}, viewProjection: I() });
  assert.deepEqual(
    d.frames.at(-1).draws.map((x) => x.id),
    [0],
  );
  p.worldMatrices[28] = 0;
  s.update(1);
  s.render({ colorView: {}, depthView: {}, viewProjection: I() });
  assert.deepEqual(
    d.frames.at(-1).draws.map((x) => x.id),
    [0, 1],
  );
  await s.whenIdle();
  s.dispose();
  assert.equal(p.disposed, false);
});
test("GPU bounds bytes participate in the exact pre-write output budget", async () => {
  const f = await setup(),
    a = f.buildAnimation(f.entry, path.join(f.root, "sized"), { webgpu: true });
  const short = path.join(f.root, "too-small");
  assert.throws(
    () => f.buildAnimation(f.entry, short, { webgpu: true, maxBytes: a.outputBytes - 1 }),
    { code: "GLTF_ANIMATION_LIMIT" },
  );
  assert.equal(fs.existsSync(short), false);
  const exact = f.buildAnimation(f.entry, path.join(f.root, "exact"), {
    webgpu: true,
    maxBytes: a.outputBytes,
  });
  assert.equal(exact.outputBytes, a.outputBytes);
});
test("the new GPU dependency does not change any emitted CPU-only package bytes", async () => {
  const f = await setup(),
    code = fs.readFileSync(path.join(f.toolkit, "build_animation.mjs"), "utf8");
  const old = code.replace(",'animation_bounds.mjs'", "");
  assert.notEqual(old, code);
  const previous = path.join(f.toolkit, "previous.mjs");
  fs.writeFileSync(previous, old);
  const { buildAnimation: before } = await import(pathToFileURL(previous));
  const a = before(f.entry, path.join(f.root, "before")),
    b = f.buildAnimation(f.entry, path.join(f.root, "after"));
  assert.equal(b.emittedFiles.includes("animation_bounds.mjs"), false);
  assert.equal(a.outputBytes, b.outputBytes);
  for (const name of a.emittedFiles)
    assert.deepEqual(
      fs.readFileSync(path.join(a.outDir, name)),
      fs.readFileSync(path.join(b.outDir, name)),
    );
});
