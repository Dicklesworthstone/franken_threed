import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// Exercise the production scene, including its pre-await material snapshot,
// reservation and failure handling. These are explicit renderer/deformer/clock
// boundaries, not shader execution or a pixel-equivalence test.
const encode = (s) => "data:text/javascript;base64," + Buffer.from(s).toString("base64");
let code = readFileSync(new URL("./animation_scene.mjs", import.meta.url), "utf8");
for (const [name, source] of [
  [
    "animation_controller.mjs",
    "export function createAnimationController(pose){return pose.controller;}",
  ],
  [
    "animation_draw_order.mjs",
    'export function createAnimationDrawOrder(){throw Error("sorting is not under test");}',
  ],
  [
    "animation_webgpu.mjs",
    "export function createGpuAnimationDeformer(d,p,g,o){return d.deform(p,g,o);}",
  ],
  [
    "animation_render.mjs",
    "export class AnimationRenderError extends Error {constructor(code,message){super(message);this.code=code;}} export function createGpuAnimationRenderer(d,o){return d.renderer(o);}",
  ],
])
  code = code.replace("'./" + name + "'", JSON.stringify(encode(source)));
const { createGpuAnimationScene } = await import(encode(code));
const MAPS = [
  "baseColorTexture",
  "metallicRoughnessTexture",
  "normalTexture",
  "emissiveTexture",
  "occlusionTexture",
];
function setup() {
  const pose = {
    version: 0,
    disposed: false,
    controller: {
      dispose() {},
      update() {
        pose.version++;
      },
    },
  };
  const materials = [],
    frames = [],
    limits = [],
    meshes = [],
    deformers = [];
  let allocated = 256,
    lit = false,
    defer = null,
    rendererCalls = 0;
  const renderer = {
    get allocatedBytes() {
      return allocated;
    },
    failed: false,
    async addMesh(gpu, material) {
      if (material.occlusionTexture && Object.hasOwn(material.occlusionTexture, "flipY"))
        throw Error("unsupported flipY");
      const vertices = gpu.vertexCount;
      allocated += vertices * (24 + Object.keys(material.mapCoordinates ?? {}).length * 8);
      if (!lit) {
        allocated += 544 + (renderer.options.environment ? 64 : 0);
        lit = true;
      }
      materials.push(material);
      const m = {
        disposed: false,
        dispose() {
          this.disposed = true;
        },
      };
      meshes.push(m);
      return m;
    },
    render(f) {
      frames.push(f);
    },
    whenIdle: async () => {},
    dispose() {
      allocated = 0;
    },
  };
  const device = {
    async renderer(options) {
      rendererCalls++;
      renderer.options = options;
      await defer?.();
      return renderer;
    },
    async deform(p, g, o) {
      limits.push(o.maxBytes);
      const gpu = {
        node: g.node,
        vertexCount: g.positions.length / 3,
        bufferBytes: 100,
        poseVersion: p.version,
        disposed: false,
        failed: false,
        update() {
          this.poseVersion = p.version;
        },
        whenIdle: async () => {},
        dispose() {
          this.disposed = true;
        },
      };
      deformers.push(gpu);
      return gpu;
    },
  };
  const drawable = {
    geometry: { node: 0, positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
    shading: "metallic-roughness",
    occlusionStrength: 0.25,
    occlusionTexture: { view: {}, sampler: {} },
    mapCoordinates: {
      occlusionTexture: {
        texCoords: new Float32Array([0, 0, 1, 0, 0, 1]),
        uvTransform: [2, 0, 0, 3, 0.25, 0.5],
      },
    },
  };
  return {
    pose,
    device,
    renderer,
    drawable,
    materials,
    frames,
    limits,
    meshes,
    deformers,
    setDefer(fn) {
      defer = fn;
    },
    get rendererCalls() {
      return rendererCalls;
    },
  };
}
const create = (f, options = {}) =>
  createGpuAnimationScene(f.device, f.pose, [f.drawable], {
    sortObjects: false,
    renderer: { environment: true },
    ...options,
  });
for (const strength of [undefined, 0, 0.25, 1])
  test(`scene forwards occlusion strength ${strength} without defaulting zero`, async () => {
    const f = setup();
    if (strength === undefined) delete f.drawable.occlusionStrength;
    else f.drawable.occlusionStrength = strength;
    const scene = await create(f);
    const m = f.materials[0];
    assert.equal(m.occlusionStrength, strength);
    assert.equal(m.occlusionTexture.view, f.drawable.occlusionTexture.view);
    assert.notEqual(m.occlusionTexture, f.drawable.occlusionTexture);
    assert.deepEqual(m.mapCoordinates.occlusionTexture.texCoords, [0, 0, 1, 0, 0, 1]);
    scene.render({ colorView: {}, depthView: {}, environment: { map: {} } });
    assert.equal(f.frames.length, 1);
    await scene.whenIdle();
    scene.dispose();
    assert.ok(f.meshes.every((m) => m.disposed) && f.deformers.every((d) => d.disposed));
    assert.equal(f.pose.disposed, false);
  });
test("all five material maps and independent coordinates coexist through scene registration", async () => {
  const f = setup();
  f.drawable.mapCoordinates = {};
  MAPS.forEach((key, i) => {
    f.drawable[key] = { view: { i }, sampler: { i } };
    f.drawable.mapCoordinates[key] = {
      texCoords: [0, 0, 1, 0, 0, 1],
      uvTransform: [1, 0, 0, 1, i, 0],
    };
  });
  const scene = await create(f);
  for (const key of MAPS) {
    assert.equal(f.materials[0][key].view, f.drawable[key].view);
    assert.deepEqual(f.materials[0].mapCoordinates[key], f.drawable.mapCoordinates[key]);
  }
  assert.equal(scene.bufferBytes, 256 + 100 + 3 * (24 + 5 * 8) + 544 + 64);
  scene.dispose();
});
test("material strength, borrowed descriptors and coordinates are captured before initialization yields", async () => {
  const f = setup();
  let resume;
  f.setDefer(
    () =>
      new Promise((r) => {
        resume = r;
      }),
  );
  const pending = create(f);
  const view = f.drawable.occlusionTexture.view;
  f.drawable.occlusionStrength = 1;
  f.drawable.occlusionTexture.view = { replaced: true };
  f.drawable.mapCoordinates.occlusionTexture.texCoords.fill(99);
  f.drawable.mapCoordinates.occlusionTexture.uvTransform.fill(99);
  resume();
  const scene = await pending;
  const m = f.materials[0];
  assert.equal(m.occlusionStrength, 0.25);
  assert.equal(m.occlusionTexture.view, view);
  assert.deepEqual(m.mapCoordinates.occlusionTexture.texCoords, [0, 0, 1, 0, 0, 1]);
  assert.deepEqual(m.mapCoordinates.occlusionTexture.uvTransform, [2, 0, 0, 3, 0.25, 0.5]);
  scene.dispose();
});
test("occlusion-only UV data reserves a surface stream before deformation allocation", async () => {
  const f = setup(),
    total = 256 + 100 + 3 * 32 + 544 + 64;
  const scene = await create(f, { maxBytes: total });
  assert.deepEqual(f.limits, [100]);
  assert.equal(scene.bufferBytes, total);
  scene.dispose();
});
test("exhausted occlusion surface budget rejects before creating a deformer", async () => {
  const f = setup();
  await assert.rejects(create(f, { maxBytes: 256 + 3 * 32 + 544 + 64 }), {
    code: "ANIMATION_SCENE_LIMIT",
  });
  assert.equal(f.deformers.length, 0);
});
test("occlusion descriptor keys survive forwarding so the renderer can reject unsupported behavior", async () => {
  const f = setup();
  f.drawable.occlusionTexture.flipY = true;
  await assert.rejects(create(f), /unsupported flipY/);
  assert.ok(f.deformers.every((d) => d.disposed));
});
for (const change of [
  (f) => {
    f.drawable.occlusionTexture = 1;
  },
  (f) => {
    f.drawable.mapCoordinates.occlusionTexture.bad = true;
  },
  (f) => {
    f.drawable.occlusionTexture = null;
  },
  (f) => {
    f.drawable.mapCoordinates.occlusionTexture.texCoords = new Float32Array(
      new SharedArrayBuffer(24),
    );
  },
])
  test("invalid occlusion descriptors/coordinate storage fail before renderer allocation", async () => {
    const f = setup();
    change(f);
    await assert.rejects(create(f));
    assert.equal(f.rendererCalls, 0);
    assert.equal(f.deformers.length, 0);
  });
