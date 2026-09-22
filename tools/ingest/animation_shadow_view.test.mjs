import assert from "node:assert/strict";
import test from "node:test";
import {
  fitAnimationShadowView as fit,
  animationShadowWorldBounds as union,
} from "./animation_shadow_view.mjs";

const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const B = (min = [-1, -1, -1], max = [1, 1, 1]) => ({ min, max });
const corners = (b) =>
  Array.from({ length: 8 }, (_, n) => [0, 1, 2].map((a) => (n & (1 << a) ? b.max[a] : b.min[a])));
const transform = (m, p) =>
  [0, 1, 2, 3].map((r) => m[r] * p[0] + m[r + 4] * p[1] + m[r + 8] * p[2] + m[r + 12]);
const close = (actual, expected, tolerance = 1e-8) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const error = { code: "ANIMATION_SHADOW_VIEW" };
function contained(m, b) {
  for (const p of corners(b)) {
    const q = transform(m.map(Math.fround), p),
      [x, y, z, w] = q;
    assert.ok(x >= -w && x <= w && y >= -w && y <= w && z >= 0 && z <= w, JSON.stringify({ p, q }));
  }
}
test("directional fitting encloses every box corner in WebGPU depth, independently of the viewer", () => {
  const box = B([-3, -2, -8], [5, 7, 4]),
    f = fit({ type: "directional" }, box);
  assert.equal(f.type, "directional");
  assert.ok(f.near > 0 && f.far > f.near);
  contained(f.viewProjection, box);
  close(transform(f.viewMatrix, f.position)[2], 0);
  close(transform(f.projectionMatrix, [0, 0, -f.near])[2], 0);
  close(transform(f.projectionMatrix, [0, 0, -f.far])[2], 1);
});
test("arbitrary light axes, near-up axes and zero-size bounds produce finite nonsingular fits", () => {
  for (const direction of [
    [0, 1, 0],
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 2, 3],
    [1e-300, 1e-300, -1e-300],
    [1e300, 1e300, 1e300],
  ]) {
    for (const box of [
      B(),
      B([42, 7, -3], [42, 7, -3]),
      B([1e8, -1e8, 2e8], [1e8 + 16, -1e8 + 32, 2e8 + 64]),
    ]) {
      const f = fit({ type: "directional", direction }, box, { padding: 0 });
      contained(f.viewProjection, box);
      assert.ok(f.viewProjection.every(Number.isFinite));
    }
  }
});
test("deterministic rotating directional fits remain enclosing through translated nonuniform bounds", () => {
  for (let n = 0; n < 100; n++) {
    const a = n * 0.31,
      box = B([100 * n - 10, -n, 0], [100 * n + 1, n + 4, 3 * n]);
    contained(
      fit({ type: "directional", direction: [Math.cos(a), Math.sin(a), -0.7] }, box).viewProjection,
      box,
    );
  }
});
test("spot lens matches the full outer cone and maps near/far to zero/one", () => {
  const f = fit(
    { type: "spot", position: [0, 0, 5], direction: [0, 0, -1], outerConeAngle: Math.PI / 6 },
    B([-1, -1, -5], [1, 1, 0]),
  );
  close(f.projectionMatrix[0], Math.sqrt(3));
  close(f.projectionMatrix[5], Math.sqrt(3));
  for (const [distance, depth] of [
    [f.near, 0],
    [f.far, 1],
  ]) {
    const q = transform(f.viewProjection, [0, 0, 5 - distance]);
    close(q[2] / q[3], depth);
    const edge = transform(f.viewProjection, [distance * Math.tan(Math.PI / 6), 0, 5 - distance]);
    close(edge[0] / edge[3], 1);
  }
  contained(f.viewProjection, B([-1, -1, -5], [1, 1, 0]));
});
test("animated spot position and direction affect its actual view, not the cone width", () => {
  const light = { type: "spot", position: [3, 2, 1], direction: [1, 0, 0] },
    f = fit(light, B([5, 1, 0], [8, 3, 2]));
  const onAxis = transform(f.viewProjection, [6, 2, 1]);
  close(onAxis[0], 0);
  close(onAxis[1], 0);
  assert.ok(onAxis[3] > 0);
  contained(f.viewProjection, B([5, 1, 0], [8, 3, 2]));
  light.position[0] = 900;
  light.direction[0] = -1;
  assert.deepEqual(f.position, [3, 2, 1]);
  assert.deepEqual(f.direction, [1, 0, 0]);
});
test("spot range caps far depth without shrinking the explicit minimum-near exclusion zone", () => {
  const f = fit({ type: "spot", position: [0, 0, 0], range: 10 }, B([-1, -1, -100], [1, 1, -1]));
  assert.equal(f.far, 10);
  const tiny = fit({ type: "spot", position: [0, 0, 0], range: 0.0015 }, B());
  assert.ok(tiny.near >= 0.001 && tiny.near < tiny.far);
  assert.throws(() => fit({ type: "spot", position: [0, 0, 0], range: 0.001 }, B()), error);
  const behind = fit({ type: "spot", position: [0, 0, 0] }, B([-1, -1, 2], [1, 1, 3]));
  assert.equal(behind.near, 0.001);
  assert.ok(behind.far > behind.near);
});
test("affine bounds union contains mirrored, sheared, scaled and translated geometry", () => {
  const a = I(),
    b = I();
  a[0] = -2;
  a[4] = 3;
  a[12] = 100;
  b[5] = 4;
  b[13] = -20;
  b[14] = 3;
  const box = B(),
    out = union([
      { bounds: box, worldMatrix: a },
      { bounds: box, worldMatrix: b },
    ]);
  for (const m of [a, b])
    for (const p of corners(box))
      transform(m, p)
        .slice(0, 3)
        .forEach((v, i) => assert.ok(v >= out.min[i] && v <= out.max[i]));
  assert.ok(out.min[0] <= -1 && out.max[0] >= 105);
  assert.ok(out.min[1] <= -24 && out.max[1] >= 1);
  assert.ok(Object.isFrozen(out) && Object.isFrozen(out.min));
});
test("cancellation-scaled arithmetic slack survives large translations", () => {
  const w = I();
  w[0] = 1e8;
  w[12] = -1e8;
  const out = union([{ bounds: B([1, 0, 0], [1, 0, 0]), worldMatrix: w }]);
  assert.ok(out.min[0] < -1 && out.max[0] > 1);
  contained(fit({ type: "directional" }, out).viewProjection, out);
});
test("snapshots own arrays and reject point lights, degenerate directions, invalid intervals and unsupported options", () => {
  const box = B(),
    f = fit({ type: "directional" }, box);
  box.min.fill(0);
  assert.deepEqual(f.bounds.min, [-1, -1, -1]);
  assert.ok(
    Object.isFrozen(f) && Object.isFrozen(f.viewMatrix) && Object.isFrozen(f.projectionMatrix),
  );
  for (const light of [
    null,
    { type: "point" },
    { type: "directional", direction: [0, 0, 0] },
    { type: "directional", direction: [NaN, 0, -1] },
    { type: "spot", position: [0, 0, 0], outerConeAngle: Math.PI / 2 },
    { type: "spot", outerConeAngle: 0 },
    { type: "spot", position: [0, 0, 0], range: Infinity },
  ])
    assert.throws(() => fit(light, B()), error);
  for (const opts of [
    { padding: -1 },
    { padding: NaN },
    { padding: 2 },
    { minNear: 0 },
    { minNear: 1e-50 },
    { foo: 1 },
  ])
    assert.throws(() => fit({ type: "directional" }, B(), opts), error);
  for (const box of [
    B([1, 1, 1], [-1, -1, -1]),
    B([-Infinity, 0, 0]),
    { min: [0, 0], max: [1, 1, 1] },
  ])
    assert.throws(() => fit({ type: "directional" }, box), error);
});
test("unbounded, detached, shared, nonaffine and overflowing world data fails before a shadow can be submitted", () => {
  const projective = I();
  projective[3] = 1;
  const huge = I();
  huge[0] = 1e300;
  for (const entries of [
    [],
    [{ bounds: B(), worldMatrix: projective }],
    [{ bounds: B(), worldMatrix: huge }],
    [{ bounds: B([-Infinity, 0, 0]), worldMatrix: I() }],
  ])
    assert.throws(() => union(entries), error);
  const detached = new Float32Array(I());
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  assert.throws(() => union([{ bounds: B(), worldMatrix: detached }]), error);
  assert.throws(
    () => fit({ type: "directional", direction: new Float32Array(new SharedArrayBuffer(12)) }, B()),
    error,
  );
});
