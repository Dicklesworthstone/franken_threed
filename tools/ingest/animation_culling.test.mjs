import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createAnimationDrawOrder } from "./animation_draw_order.mjs";

// Production scene + bounds + ordering. Only the controller, GPU deformer and
// renderer boundaries are recording doubles. No shader or pixel emulation.
const moduleURL = (s) => "data:text/javascript;base64," + Buffer.from(s).toString("base64");
const stubs = moduleURL(`
export class AnimationRenderError extends Error{constructor(code,message){super(message);this.code=code;}}
export function createAnimationController(pose){return {update(dt){pose.advance?.(dt);pose.version++;},dispose(){}};}
export async function createGpuAnimationDeformer(device,pose,geometry){return device.deformer(pose,geometry);}
export async function createGpuAnimationRenderer(device,options){return device.renderer(options);}
`);
let source = readFileSync(new URL("./animation_scene.mjs", import.meta.url), "utf8");
for (const name of [
  "animation_controller.mjs",
  "animation_webgpu.mjs",
  "animation_render.mjs",
  "animation_draw_order.mjs",
]) {
  const quoted = "'./" + name + "'";
  assert.equal(source.split(quoted).length, 2);
  source = source.replace(
    quoted,
    JSON.stringify(
      name === "animation_draw_order.mjs" ? new URL("./" + name, import.meta.url).href : stubs,
    ),
  );
}
const { createGpuAnimationScene } = await import(moduleURL(source));
const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function pose(n, targets = 0) {
  return {
    nodeCount: n,
    version: 0,
    disposed: false,
    instances: [],
    jointMatrices: new Float64Array(),
    morphOffsets: Uint32Array.from(Array.from({ length: n + 1 }, (_, i) => i * targets)),
    morphWeights: new Float64Array(n * targets),
    worldMatrices: new Float64Array(Array.from({ length: n }, I).flat()),
  };
}
function drawable(node, targets = 0, alphaMode = "OPAQUE") {
  return {
    geometry: {
      node,
      positions: [-0.1, -0.1, 0.5, 0.1, -0.1, 0.5, 0, 0.1, 0.5],
      morphTargets: Array.from({ length: targets }, () => ({
        positions: [1, 0, 0, 1, 0, 0, 1, 0, 0],
      })),
    },
    alphaMode,
  };
}
function device() {
  const deformers = [],
    meshes = [],
    frames = [];
  let updates = 0;
  const renderer = {
    allocatedBytes: 0,
    disposed: false,
    failed: false,
    async addMesh(gpu, material) {
      const m = {
        gpu,
        material,
        id: meshes.length,
        disposed: false,
        dispose() {
          this.disposed = true;
        },
      };
      meshes.push(m);
      return m;
    },
    render(frame) {
      if (this.failNext) {
        this.failNext = false;
        throw Error("bad attachment");
      }
      frames.push({ ...frame, draws: frame.draws.slice() });
    },
    async whenIdle() {
      if (this.terminal) {
        this.failed = true;
        throw Error("device failed");
      }
    },
    dispose() {
      this.disposed = true;
    },
  };
  const d = {
    deformers,
    meshes,
    frames,
    output: renderer,
    get updates() {
      return updates;
    },
    async renderer() {
      return renderer;
    },
    async deformer(p, g) {
      const gpu = {
        node: g.node,
        worldMatrix: new Float64Array(p.worldMatrices.slice(g.node * 16, g.node * 16 + 16)),
        bufferBytes: 120,
        poseVersion: p.version,
        disposed: false,
        failed: false,
        update() {
          if (d.failNode === g.node) throw Error("upload failed");
          updates++;
          this.poseVersion = p.version;
          this.worldMatrix.set(p.worldMatrices.slice(g.node * 16, g.node * 16 + 16));
        },
        async whenIdle() {},
        dispose() {
          this.disposed = true;
        },
      };
      deformers.push(gpu);
      if (d.afterCreate) await d.afterCreate(gpu);
      return gpu;
    },
  };
  return d;
}
const frame = () => ({ colorView: {}, depthView: {}, viewProjection: I() });
const ids = (d) => d.frames.at(-1).draws.map((x) => x.id);

test("implicit offscreen draws are omitted and summary work is independent of vertices", async () => {
  const p = pose(3),
    d = device();
  p.worldMatrices[28] = 20;
  p.worldMatrices[44] = -20;
  const s = await createGpuAnimationScene(d, p, [drawable(0), drawable(1), drawable(2)], {
    frustumCulling: true,
  });
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  assert.equal(s.boundsBytes, 144);
  assert.deepEqual(s.cullingStats, {
    poseVersion: 0,
    testedMeshes: 3,
    culledMeshes: 2,
    submittedDraws: 1,
  });
  assert.equal(d.updates, 0);
  s.dispose();
  assert.equal(s.boundsBytes, 0);
  assert.equal(p.disposed, false);
});
test("a 1000-mesh view submits only its ten potentially visible meshes", async () => {
  const n = 1000,
    p = pose(n),
    d = device();
  for (let i = 10; i < n; i++) p.worldMatrices[i * 16 + 12] = i * 3;
  const s = await createGpuAnimationScene(
    d,
    p,
    Array.from({ length: n }, (_, i) => drawable(i)),
    { frustumCulling: true, maxMeshes: n },
  );
  s.render(frame());
  assert.deepEqual(
    ids(d),
    Array.from({ length: 10 }, (_, i) => i),
  );
  assert.equal(s.cullingStats.culledMeshes, 990);
  assert.equal(s.boundsBytes, 48000);
  s.update(0.016);
  assert.equal(d.updates, n, "Culling does not secretly change deformation/update timing");
  s.dispose();
});
test("a new camera is applied every render and shares one snapshot with submission", async () => {
  const p = pose(2),
    d = device();
  p.worldMatrices[28] = 10;
  const s = await createGpuAnimationScene(d, p, [drawable(0), drawable(1)], {
    frustumCulling: true,
  });
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  const f = frame();
  f.viewProjection[12] = -10;
  s.render(f);
  assert.deepEqual(ids(d), [1]);
  assert.notEqual(d.frames.at(-1).viewProjection, f.viewProjection);
  f.viewProjection[12] = 999;
  assert.equal(d.frames.at(-1).viewProjection[12], -10);
  assert.equal(p.version, 0);
  s.dispose();
});
test("updated morph weights move a mesh into and out of view without scanning geometry again", async () => {
  const p = pose(1, 1),
    d = device(),
    item = drawable(0, 1);
  p.worldMatrices[12] = 10;
  const s = await createGpuAnimationScene(d, p, [item], { frustumCulling: true });
  s.render(frame());
  assert.deepEqual(ids(d), []);
  Object.defineProperty(item.geometry, "positions", {
    get() {
      throw Error("Source vertices accessed again");
    },
  });
  Object.defineProperty(item.geometry, "morphTargets", {
    get() {
      throw Error("Source targets accessed again");
    },
  });
  p.advance = () => {
    p.morphWeights[0] = -10;
  };
  s.update(1);
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  assert.equal(s.cullingStats.poseVersion, p.version);
  p.advance = () => {
    p.morphWeights[0] = 10;
  };
  s.update(1);
  s.render(frame());
  assert.deepEqual(ids(d), []);
  s.dispose();
});
test("skin palettes and node transforms both participate after each upload", async () => {
  const p = pose(1),
    d = device(),
    item = drawable(0);
  p.instances = [{ node: 0, offset: 16, jointCount: 1 }];
  p.jointMatrices = new Float64Array([...I(), ...I()]);
  Object.assign(item.geometry, {
    joints: Array(12).fill(0),
    weights: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
  });
  const s = await createGpuAnimationScene(d, p, [item], { frustumCulling: true });
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  p.jointMatrices[28] = 5;
  p.version++;
  s.upload();
  s.render(frame());
  assert.deepEqual(ids(d), []);
  p.worldMatrices[12] = -5;
  p.version++;
  s.upload();
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  s.dispose();
});
test("culling preserves opaque/mask order and stable back-to-front transparent sorting", async () => {
  const p = pose(6),
    d = device(),
    items = ["BLEND", "OPAQUE", "BLEND", "MASK", "BLEND", "BLEND"].map((mode, i) =>
      drawable(i, 0, mode),
    );
  p.worldMatrices[14] = 0.1;
  p.worldMatrices[46] = 0.3;
  p.worldMatrices[78] = 0.3;
  p.worldMatrices[92] = 100;
  const s = await createGpuAnimationScene(d, p, items, { frustumCulling: true });
  s.render(frame());
  assert.deepEqual(ids(d), [1, 3, 2, 4, 0]);
  assert.equal(s.cullingStats.culledMeshes, 1);
  s.dispose();
});
test("sorting can be disabled while retaining culling and original survivor order", async () => {
  const p = pose(3),
    d = device();
  p.worldMatrices[28] = 100;
  const s = await createGpuAnimationScene(
    d,
    p,
    [drawable(0, 0, "BLEND"), drawable(1), drawable(2)],
    { frustumCulling: true, sortObjects: false },
  );
  s.render(frame());
  assert.deepEqual(ids(d), [0, 2]);
  s.dispose();
});
test("explicit draws, duplicates, overrides and empty clears bypass automatic filtering", async () => {
  const p = pose(2),
    d = device();
  p.worldMatrices[28] = 100;
  const s = await createGpuAnimationScene(d, p, [drawable(0), drawable(1)], {
    frustumCulling: true,
  });
  const override = { mesh: s.draws[1], worldMatrix: I() };
  s.render({ ...frame(), draws: [override, s.draws[1], s.draws[1]] });
  assert.equal(d.frames.at(-1).draws[0], override);
  assert.equal(d.frames.at(-1).draws[1], s.draws[1]);
  assert.deepEqual(s.cullingStats, {
    poseVersion: 0,
    testedMeshes: 0,
    culledMeshes: 0,
    submittedDraws: 3,
  });
  s.render({ ...frame(), draws: [] });
  assert.deepEqual(ids(d), []);
  s.dispose();
});
test("disabled culling scans nothing and does not apply culling-only limits", async () => {
  const p = pose(1),
    d = device();
  delete p.morphOffsets;
  p.worldMatrices[12] = 100;
  const s = await createGpuAnimationScene(d, p, [drawable(0)], {
    maxBoundsBytes: 0,
    maxBoundsComponents: 0,
  });
  s.render(frame());
  assert.deepEqual(ids(d), [0]);
  assert.equal(s.boundsBytes, 0);
  assert.equal(s.cullingStats, null);
  s.dispose();
});
test("invalid options, aggregate summary and scan limits release constructed resources", async () => {
  for (const config of [
    { frustumCulling: 1 },
    { frustumCulling: true, maxBoundsBytes: 0 },
    { frustumCulling: true, maxBoundsBytes: 95 },
    { frustumCulling: true, maxBoundsComponents: 17 },
  ]) {
    const p = pose(2),
      d = device();
    await assert.rejects(createGpuAnimationScene(d, p, [drawable(0), drawable(1)], config));
    assert.ok(d.deformers.every((g) => g.disposed));
    assert.ok(d.meshes.every((m) => m.disposed));
    assert.equal(p.disposed, false);
  }
});
test("stale pose renders are rejected until upload refreshes geometry and bounds together", async () => {
  const p = pose(1, 1),
    d = device(),
    s = await createGpuAnimationScene(d, p, [drawable(0, 1)], { frustumCulling: true });
  s.render(frame());
  const before = s.cullingStats;
  p.morphWeights[0] = 100;
  p.version++;
  assert.throws(() => s.render(frame()), { code: "ANIMATION_SCENE_STALE" });
  assert.equal(d.frames.length, 1);
  assert.equal(s.cullingStats, before);
  s.upload();
  s.render(frame());
  assert.deepEqual(ids(d), []);
  s.dispose();
});
test("matrix errors and render rejection leave previous stats and resources usable", async () => {
  const p = pose(1),
    d = device(),
    s = await createGpuAnimationScene(d, p, [drawable(0)], { frustumCulling: true });
  s.render(frame());
  const old = s.cullingStats;
  assert.throws(() => s.render({ ...frame(), viewProjection: Array(16).fill(NaN) }));
  assert.equal(s.cullingStats, old);
  assert.equal(s.failed, false);
  d.output.failNext = true;
  assert.throws(() => s.render(frame()), /bad attachment/);
  assert.equal(s.cullingStats, old);
  s.render(frame());
  assert.equal(d.frames.length, 2);
  s.dispose();
});
test("camera values are sampled once and getter reentry cannot partially render", async () => {
  const p = pose(1),
    d = device(),
    s = await createGpuAnimationScene(d, p, [drawable(0)], { frustumCulling: true });
  let reads = 0;
  const vp = I();
  for (let i = 0; i < 16; i++) {
    const value = vp[i];
    Object.defineProperty(vp, i, {
      get() {
        reads++;
        return value;
      },
    });
  }
  s.render({ ...frame(), viewProjection: vp });
  assert.equal(reads, 16);
  const bad = I();
  Object.defineProperty(bad, 0, {
    get() {
      s.dispose();
      return 1;
    },
  });
  assert.throws(() => s.render({ ...frame(), viewProjection: bad }), {
    code: "ANIMATION_SCENE_REENTRANT",
  });
  assert.equal(d.frames.length, 1);
  assert.equal(s.disposed, false);
  s.dispose();
});
test("later upload failure terminates the group and releases every bounds summary", async () => {
  const p = pose(2),
    d = device(),
    s = await createGpuAnimationScene(d, p, [drawable(0), drawable(1)], { frustumCulling: true });
  d.failNode = 1;
  assert.throws(() => s.update(1), /upload failed/);
  assert.equal(s.failed, true);
  assert.equal(s.boundsBytes, 0);
  assert.ok(d.deformers.every((g) => g.disposed));
  assert.ok(d.meshes.every((m) => m.disposed));
  assert.equal(p.disposed, false);
  s.dispose();
});
test("terminal asynchronous renderer failure releases culling summaries too", async () => {
  const p = pose(1),
    d = device(),
    s = await createGpuAnimationScene(d, p, [drawable(0)], { frustumCulling: true });
  d.output.terminal = true;
  await assert.rejects(s.whenIdle(), /device failed/);
  assert.equal(s.boundsBytes, 0);
  assert.equal(s.failed, true);
  s.dispose();
});
test("standalone ordering owns only bounds, rejects stale updates and reentry", () => {
  const p = pose(1),
    g = { worldMatrix: I(), poseVersion: 0 },
    mesh = {},
    sorter = createAnimationDrawOrder(
      [{ mesh, deformer: g, geometry: drawable(0).geometry, alphaMode: "OPAQUE" }],
      { pose: p, frustumCulling: true },
    );
  assert.deepEqual(sorter.order(I()), [mesh]);
  p.version++;
  assert.throws(() => sorter.order(I()));
  g.poseVersion = p.version;
  sorter.updateBounds();
  assert.deepEqual(sorter.order(I()), [mesh]);
  const camera = I();
  Object.defineProperty(camera, 0, {
    get() {
      sorter.updateBounds();
      return 1;
    },
  });
  assert.throws(() => sorter.order(camera));
  sorter.dispose();
  assert.equal(sorter.boundsBytes, 0);
  assert.equal(p.disposed, false);
  assert.equal(g.disposed, undefined);
});
