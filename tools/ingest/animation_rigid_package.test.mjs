import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

// Production builder, scene and rigid pool. The asset decoder, pose sampler,
// ordinary compute, draw-order and material renderer are explicit boundaries.
// This suite verifies package closure/execution, not asset decoding or pixels.
async function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-rigid-package-")),
    tools = path.join(root, "tools");
  fs.mkdirSync(tools);
  for (const name of ["build_animation.mjs", "animation_scene.mjs", "animation_rigid_geometry.mjs"])
    fs.copyFileSync(new URL("./" + name, import.meta.url), path.join(tools, name));
  const sources = {
    "animation_gltf.mjs":
      "export function decodeGltfAnimation(){return {format:'f3d-animation-v1',nodes:[],ignoredChannels:[]};}",
    "gltf_instancing.mjs":
      "export function expandGltfInstances(json){return {json,instanceCount:0};}",
    "animation_runtime.mjs":
      "export class AnimationPoseError extends Error{constructor(code,text){super(text);this.code=code;}} export function createAnimationPlayer(){return {nodeCount:0,clips:[],instances:[],morphWeights:[],dispose(){}};}",
    "animation_controller.mjs":
      "export function createAnimationController(p){return p.controller;}",
    "animation_deformer.mjs":
      "export function createAnimationDeformer(){throw Error('CPU deformation is not under test');}",
    "animation_webgpu.mjs":
      "export function createGpuAnimationDeformer(){throw Error('Compute deformation is not under test');}",
    "animation_render.mjs":
      "export class AnimationRenderError extends Error{constructor(code,text){super(text);this.code=code;}} export function createGpuAnimationRenderer(d,o){return d.renderer(o);}",
    "animation_draw_order.mjs":
      "export function createAnimationDrawOrder(){throw Error('Sorting is not under test');}",
    "animation_bounds.mjs": "export {};",
    "animation_shadow.mjs":
      "export function createGpuAnimationShadowMap(){throw Error('Shadows are not under test');}",
    "animation_shadow_receiver.mjs": "export {};",
    "animation_scene_shadow.mjs": "export {};",
    "animation_shadow_view.mjs":
      "export function fitAnimationShadowView(){} export function animationShadowWorldBounds(){}",
  };
  for (const [name, text] of Object.entries(sources))
    fs.writeFileSync(path.join(tools, name), text);
  const entry = path.join(root, "model.gltf");
  fs.writeFileSync(entry, JSON.stringify({ asset: { version: "2.0" } }));
  const { buildAnimation } = await import(
    pathToFileURL(path.join(tools, "build_animation.mjs")).href
  );
  return { root, tools, entry, build: buildAnimation };
}
const contents = (dir) =>
  Object.fromEntries(
    fs
      .readdirSync(dir)
      .sort()
      .map((name) => [name, fs.readFileSync(path.join(dir, name))]),
  );
test("rigid pool source and public exports are emitted only when requested", async () => {
  const f = await fixture(),
    normal = path.join(f.root, "normal"),
    rigid = path.join(f.root, "rigid");
  const a = f.build(f.entry, normal, { webgpu: true }),
    b = f.build(f.entry, rigid, { webgpu: true, rigidGeometry: true });
  assert.equal(a.gpuRigidGeometry, undefined);
  assert.equal(
    b.gpuRigidGeometry,
    "shared-immutable-f32-vertices; opt-in scene rigidGeometry:true",
  );
  assert.deepEqual(
    b.emittedFiles.filter((x) => !a.emittedFiles.includes(x)),
    ["animation_rigid_geometry.mjs"],
  );
  assert.deepEqual(
    fs.readFileSync(path.join(rigid, "animation_rigid_geometry.mjs")),
    fs.readFileSync(path.join(f.tools, "animation_rigid_geometry.mjs")),
  );
  const module = await import(pathToFileURL(path.join(rigid, "gpu_playback.mjs")).href);
  assert.equal(typeof module.createGpuRigidGeometryPool, "function");
  assert.equal(typeof module.canUseRigidAnimationGeometry, "function");
});
test("omitted and false preserve CPU/GPU package bytes and do not read the optional module", async () => {
  const f = await fixture();
  fs.renameSync(
    path.join(f.tools, "animation_rigid_geometry.mjs"),
    path.join(f.tools, "rigid-retained.mjs"),
  );
  for (const webgpu of [false, true]) {
    const a = path.join(f.root, "a" + webgpu),
      b = path.join(f.root, "b" + webgpu);
    f.build(f.entry, a, { webgpu });
    f.build(f.entry, b, { webgpu, rigidGeometry: false });
    assert.deepEqual(contents(a), contents(b));
  }
});
for (const options of [
  { rigidGeometry: true },
  { webgpu: true, rigidGeometry: "true" },
  { webgpu: true, rigidGeometry: null },
])
  test(`invalid rigid package options fail before output creation: ${JSON.stringify(options)}`, async () => {
    const f = await fixture(),
      out = path.join(f.root, "out");
    assert.throws(() => f.build(f.entry, out, options), TypeError);
    assert.equal(fs.existsSync(out), false);
  });
test("generated rigid package enforces its exact complete output byte budget", async () => {
  const f = await fixture(),
    a = path.join(f.root, "a"),
    b = path.join(f.root, "b"),
    c = path.join(f.root, "c");
  const first = f.build(f.entry, a, { webgpu: true, rigidGeometry: true });
  assert.equal(
    first.outputBytes,
    Object.values(contents(a)).reduce((n, b) => n + b.length, 0),
  );
  f.build(f.entry, b, { webgpu: true, rigidGeometry: true, maxBytes: first.outputBytes });
  assert.deepEqual(contents(a), contents(b));
  assert.throws(
    () =>
      f.build(f.entry, c, { webgpu: true, rigidGeometry: true, maxBytes: first.outputBytes - 1 }),
    { code: "GLTF_ANIMATION_LIMIT" },
  );
  assert.equal(fs.existsSync(c), false);
});
test("relocated GPU entry executes the real scene and shared buffer pool without its source toolkit", async () => {
  const f = await fixture(),
    built = path.join(f.root, "built"),
    out = path.join(f.root, "relocated");
  f.build(f.entry, built, { webgpu: true, rigidGeometry: true });
  fs.renameSync(built, out);
  fs.renameSync(f.tools, f.tools + "-retained");
  assert.equal(fs.existsSync(f.tools), false);
  const module = await import(pathToFileURL(path.join(out, "gpu_playback.mjs")).href);
  const pose = {
    nodeCount: 2,
    version: 0,
    disposed: false,
    instances: [],
    morphOffsets: new Uint32Array(3),
    worldMatrices: new Float64Array(32),
    controller: {
      update() {
        pose.version++;
        pose.worldMatrices[28] = 10;
      },
      dispose() {},
    },
  };
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  pose.worldMatrices.set(I);
  pose.worldMatrices.set(I, 16);
  const buffers = [],
    frames = [];
  let disposed = false;
  const device = {
    limits: { maxBufferSize: 2 ** 20 },
    lost: new Promise(() => {}),
    pushErrorScope() {},
    popErrorScope: async () => null,
    queue: { onSubmittedWorkDone: async () => {} },
    createBuffer(desc) {
      const data = new ArrayBuffer(desc.size);
      const b = {
        ...desc,
        destroyed: 0,
        getMappedRange: () => data,
        unmap() {},
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    renderer(o) {
      return {
        get allocatedBytes() {
          return disposed ? 0 : 256 * o.maxDraws;
        },
        failed: false,
        async addMesh(gpu, material) {
          return { gpu, material, dispose() {} };
        },
        render(frame) {
          frames.push(frame);
        },
        whenIdle: async () => {},
        dispose() {
          disposed = true;
        },
      };
    },
  };
  const drawables = [0, 1].map((node) => ({
    geometry: { node, positions: [0, 0, 0, 1, 0, 0, 0, 1, 0] },
  }));
  const scene = await module.createGpuAnimationScene(device, pose, drawables, {
    sortObjects: false,
    rigidGeometry: true,
    maxBytes: 632,
  });
  assert.equal(buffers.length, 1);
  assert.equal(scene.deformers[0].vertexBuffer, scene.deformers[1].vertexBuffer);
  scene.update(1);
  scene.render({ viewProjection: I });
  assert.equal(scene.deformers[1].worldMatrix[12], 10);
  assert.equal(frames.length, 1);
  assert.equal(scene.bufferBytes, 632);
  await scene.whenIdle();
  scene.dispose();
  assert.equal(buffers[0].destroyed, 1);
  assert.equal(pose.disposed, false);
});
