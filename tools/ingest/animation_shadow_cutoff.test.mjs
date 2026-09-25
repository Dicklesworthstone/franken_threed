import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Exercise the production shadow-map wrapper with a recorded renderer/device
// boundary. This suite proves draw forwarding and publication, not WGSL pixels.
const key = Symbol.for("f3d.shadow-cutoff-test");
const moduleUrl = (text) => "data:text/javascript;base64," + Buffer.from(text).toString("base64");
const rendererUrl = moduleUrl(`
export class AnimationRenderError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
export async function createGpuAnimationRenderer() {
  return globalThis[Symbol.for("f3d.shadow-cutoff-test")];
}
`);
const source = await readFile(new URL("./animation_shadow.mjs", import.meta.url), "utf8");
assert.ok(source.includes('"./animation_render.mjs"'));
const { createGpuAnimationShadowMap } = await import(moduleUrl(
  source.replace('"./animation_render.mjs"', JSON.stringify(rendererUrl)),
));
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
async function setup() {
  const renderer = {
    version: 0, allocatedBytes: 256, failed: false, disposed: false,
    bindings: [], frames: [], attempts: [], error: null,
    async addMesh(gpu, material) {
      const mesh = { gpu, material, vertexCount: 3, indexCount: 0, disposed: false,
        dispose() { this.disposed = true; } };
      this.bindings.push(mesh);
      return mesh;
    },
    render(frame) {
      this.attempts.push(frame);
      if (this.error) throw this.error;
      this.frames.push(frame);
      this.version++;
    },
    async whenIdle() {},
    dispose() { this.disposed = true; },
  };
  const device = {
    limits: { maxTextureDimension2D: 1024 }, lost: new Promise(() => {}),
    pushErrorScope() {}, async popErrorScope() { return null; },
    createTexture() { return { createView: () => ({}), destroy() {} }; },
    createSampler: () => ({}),
  };
  globalThis[key] = renderer;
  const map = await createGpuAnimationShadowMap(device, { width: 8, maxDraws: 4 });
  const gpu = { worldMatrix: identity(), version: 0, poseVersion: 0, disposed: false, failed: false };
  const material = { alphaMode: "MASK", alphaCutoff: 0.3 };
  const mesh = await map.addMesh(gpu, material);
  const render = (draws) => map.render({ viewProjection: identity(), draws });
  return { renderer, device, map, gpu, material, mesh, render };
}

test("source MASK draws forward live cutoffs without replacing caster bindings", async () => {
  const { renderer, map, material, mesh, render } = await setup();
  try {
    render([{ mesh, alphaCutoff: 0.3 }]);
    render([{ mesh, alphaCutoff: 0.7 }]);
    assert.deepEqual(renderer.frames.map(f => f.draws[0].alphaCutoff), [0.3, 0.7]);
    assert.equal(renderer.bindings.length, 1);
    assert.equal(renderer.frames[1].draws[0].mesh, renderer.bindings[0]);
    assert.equal(material.alphaCutoff, 0.3);
    assert.equal(map.version, 2);
    await map.whenIdle();
  } finally { map.dispose(); }
});

test("shared caster uses retain independent per-draw cutoffs, UVs and ranges", async () => {
  const { renderer, map, mesh, render } = await setup();
  try {
    const uv = new Float32Array([2, 0, 0, 2, 0.25, 0.5]);
    const color = new Float32Array([1, 1, 1, 0.5]);
    const a = { mesh, alphaCutoff: 0.2, first: 0, count: 1, uvTransform: uv, baseColor: color };
    const b = { mesh, alphaCutoff: 0.8, first: 1, count: 2 };
    render([a, b]);
    a.alphaCutoff = 0.9; uv.fill(0); color.fill(0);
    const [first, second] = renderer.frames[0].draws;
    assert.deepEqual([first.alphaCutoff, second.alphaCutoff], [0.2, 0.8]);
    assert.deepEqual([first.first, first.count, second.first, second.count], [0, 1, 1, 2]);
    assert.deepEqual(first.uvTransform, [2, 0, 0, 2, 0.25, 0.5]);
    assert.deepEqual(first.baseColor, [1, 1, 1, 0.5]);
  } finally { map.dispose(); }
});

test("zero and one cutoffs survive forwarding while omitted defaults stay omitted", async () => {
  const { renderer, map, mesh, render } = await setup();
  try {
    render([{ mesh, alphaCutoff: 0 }, { mesh, alphaCutoff: 1 }, mesh,
      { mesh, alphaCutoff: undefined }]);
    const draws = renderer.frames[0].draws;
    assert.deepEqual(draws.slice(0, 2).map(d => d.alphaCutoff), [0, 1]);
    for (const draw of draws.slice(2)) assert.equal(Object.hasOwn(draw, "alphaCutoff"), false);
  } finally { map.dispose(); }
});

test("renderer validation failures preserve the previous published shadow snapshot", async () => {
  const { renderer, device, map, mesh, render } = await setup();
  try {
    render([{ mesh, alphaCutoff: 0.3 }]);
    const prior = map.sample(device);
    const error = new Error("renderer cutoff validation");
    renderer.error = error;
    assert.throws(() => render([{ mesh, alphaCutoff: 2 }]), e => e === error);
    assert.equal(renderer.attempts.at(-1).draws[0].alphaCutoff, 2);
    assert.equal(map.sample(device), prior);
    assert.equal(map.version, 1);
    renderer.error = null;
    render([{ mesh, alphaCutoff: 0.5 }]);
    assert.equal(map.version, 2);
    assert.notEqual(map.sample(device), prior);
  } finally { map.dispose(); }
});

test("adding cutoff support does not admit unknown fields or foreign caster handles", async () => {
  const { renderer, map, mesh, render } = await setup();
  try {
    assert.throws(() => render([{ mesh, alphaCutoff: 0.5, alphaTest: 0.5 }]), /Unsupported shadow field: alphaTest/);
    assert.throws(() => render([{ mesh: {}, alphaCutoff: 0.5 }]), /Caster does not belong/);
    assert.equal(renderer.attempts.length, 0);
    render([mesh]);
    assert.equal(renderer.frames.length, 1);
  } finally { map.dispose(); }
});
