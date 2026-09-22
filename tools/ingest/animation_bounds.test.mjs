import assert from "node:assert/strict";
import test from "node:test";
import { animationBoundsVisible, createAnimationBounds } from "./animation_bounds.mjs";

const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function fixture({ vertices = 3, targets = 0, joints = 0, influences = 4, node = 0 } = {}) {
  const pose = {
    nodeCount: node + 1,
    version: 0,
    disposed: false,
    instances: joints ? [{ node, offset: 16, jointCount: joints }] : [],
    morphOffsets: Uint32Array.from(
      Array(node + 1)
        .fill(0)
        .concat(targets),
    ),
    morphWeights: new Float64Array(targets),
    jointMatrices: new Float64Array([I(), ...Array.from({ length: joints }, I)].flat()),
  };
  const geometry = {
    node,
    positions: Float32Array.from(Array.from({ length: vertices }, (_, i) => [i, 0, 0]).flat()),
    morphTargets: Array.from({ length: targets }, () => ({
      positions: new Float32Array(vertices * 3),
    })),
  };
  if (joints)
    Object.assign(geometry, {
      influences,
      joints: new Uint32Array(vertices * influences),
      weights: Float64Array.from(
        Array.from({ length: vertices }, () => [1, ...Array(influences - 1).fill(0)]).flat(),
      ),
    });
  return { pose, geometry };
}
const contains = (b, p) => {
  for (let a = 0; a < 3; a++)
    assert.ok(
      p[a] >= b.min[a] && p[a] <= b.max[a],
      `${p[a]} outside [${b.min[a]}, ${b.max[a]}], axis ${a}`,
    );
};
function reference(pose, g, { flush = false, fused = false, reverse = false } = {}) {
  const F = (x) => {
    const y = Math.fround(x);
    return flush && Math.abs(y) < 2 ** -126 ? 0 : y;
  };
  const out = [];
  for (let v = 0; v < g.positions.length / 3; v++) {
    const p = Array.from(g.positions.slice(v * 3, v * 3 + 3), F);
    for (let t = 0; t < g.morphTargets.length; t++) {
      const w = F(pose.morphWeights[t]);
      if (!w) continue;
      for (let a = 0; a < 3; a++) {
        const delta = F(g.morphTargets[t].positions?.[v * 3 + a] ?? 0);
        p[a] = F(p[a] + (fused ? w * delta : F(w * delta)));
      }
    }
    if (pose.instances.length) {
      const n = g.influences,
        skinned = [0, 0, 0];
      for (let x = 0; x < n; x++) {
        const k = reverse ? n - x - 1 : x,
          at = v * n + k,
          w = F(g.weights[at]);
        if (!w) continue;
        const offset = pose.instances[0].offset + g.joints[at] * 16;
        for (let a = 0; a < 3; a++) {
          const terms = [0, 1, 2].map((c) => F(pose.jointMatrices[offset + c * 4 + a]) * p[c]);
          terms.push(F(pose.jointMatrices[offset + 12 + a]));
          if (reverse) terms.reverse();
          const transformed = fused
            ? F(terms.reduce((x, y) => x + y, 0))
            : terms.map(F).reduce((x, y) => F(x + y), 0);
          skinned[a] = F(skinned[a] + (fused ? w * transformed : F(w * transformed)));
        }
      }
      out.push(skinned);
    } else out.push(p);
  }
  return out;
}
const box = (min, max) => ({ min, max });
const perspective = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -10 / 9, -1, 0, 0, -10 / 9, 0];

test("static geometry is scanned once and retains only six summary values", () => {
  const f = fixture({ vertices: 10000 });
  const b = createAnimationBounds(f.pose, f.geometry);
  assert.deepEqual(b.snapshot.min, [0, 0, 0]);
  assert.deepEqual(b.snapshot.max, [9999, 0, 0]);
  assert.equal(b.byteLength, 48);
  assert.equal(b.sourceComponents, 30000);
  f.geometry.positions.fill(999999);
  f.geometry.morphTargets.push({ positions: [] });
  const before = b.snapshot;
  f.pose.version++;
  b.update();
  assert.equal(b.snapshot.max[0], 9999);
  assert.equal(before.poseVersion, 0);
  assert.equal(b.snapshot.poseVersion, 1);
  assert.ok(Object.isFrozen(before.min));
  b.dispose();
  b.dispose();
  assert.equal(b.byteLength, 0);
  assert.equal(f.pose.disposed, false);
});
test("signed and greater-than-one morph weights update without retaining source targets", () => {
  const f = fixture({ targets: 2 });
  f.geometry.morphTargets[0].positions.set([1, 2, 0, 2, 3, 0, -1, -1, 0]);
  f.geometry.morphTargets[1].positions.set([0, 1, 1, 0, 2, 1, 0, -2, 1]);
  const b = createAnimationBounds(f.pose, f.geometry);
  f.pose.morphWeights.set([-2, 3]);
  f.pose.version++;
  for (const p of reference(f.pose, f.geometry)) contains(b.update(), p);
  const snapshot = b.snapshot;
  f.geometry.morphTargets.length = 0;
  assert.deepEqual(b.update().min, snapshot.min);
  assert.deepEqual(b.update().max, snapshot.max);
});
test("normal-only morph targets do not move position bounds", () => {
  const f = fixture({ targets: 1 });
  f.geometry.morphTargets = [{ normals: [1, 2, 3] }];
  f.pose.morphWeights[0] = 4;
  const b = createAnimationBounds(f.pose, f.geometry);
  for (const p of [
    [0, 0, 0],
    [2, 0, 0],
  ])
    contains(b.snapshot, p);
  assert.equal(b.sourceComponents, 9);
  assert.ok(b.snapshot.max[0] < 2.00001);
});
test("instance-local palette offset and nonunit weight sums are honored", () => {
  const f = fixture({ joints: 2, influences: 2, node: 1 });
  f.geometry.weights.set([0.50002, 0.50002, 0.50002, 0.50002, 0.50002, 0.50002]);
  f.geometry.joints.set([0, 1, 0, 1, 0, 1]);
  f.pose.jointMatrices[12] = 99999;
  f.pose.jointMatrices[28] = 10;
  f.pose.jointMatrices[44] = 20;
  const b = createAnimationBounds(f.pose, f.geometry);
  for (const p of reference(f.pose, f.geometry)) contains(b.snapshot, p);
  assert.ok(b.snapshot.min[0] > 9.99);
  assert.ok(b.snapshot.max[0] < 22.01);
  assert.equal(b.byteLength, 56);
});
test("unused joint transforms do not expand the hull", () => {
  const f = fixture({ joints: 32 });
  f.pose.jointMatrices[28] = 10;
  for (let j = 1; j < 32; j++) f.pose.jointMatrices[(j + 1) * 16 + 12] = 1e10;
  const b = createAnimationBounds(f.pose, f.geometry);
  assert.equal(b.byteLength, 52);
  assert.ok(b.snapshot.max[0] < 12.01);
});
test("morph-plus-skin bounds cover many independent f32 evaluations, including 32 influences", () => {
  let seed = 5121;
  const R = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  let evaluated = 0;
  for (const influences of [1, 2, 4, 8, 16, 32]) {
    const f = fixture({ vertices: 41, targets: 3, joints: 7, influences });
    for (let i = 0; i < f.geometry.positions.length; i++)
      f.geometry.positions[i] = (R() - 0.5) * 40;
    for (const t of f.geometry.morphTargets)
      for (let i = 0; i < t.positions.length; i++) t.positions[i] = (R() - 0.5) * 3;
    for (let v = 0; v < 41; v++) {
      let sum = 0;
      const weights = Array.from({ length: influences }, () => {
        const w = R();
        sum += w;
        return w;
      });
      for (let k = 0; k < influences; k++) {
        f.geometry.weights[v * influences + k] = weights[k] / sum;
        f.geometry.joints[v * influences + k] = Math.floor(R() * 7);
      }
    }
    const b = createAnimationBounds(f.pose, f.geometry);
    for (let frame = 0; frame < 20; frame++) {
      for (let t = 0; t < 3; t++) f.pose.morphWeights[t] = (R() - 0.5) * 5;
      for (let j = 0; j < 7; j++) {
        const offset = (j + 1) * 16;
        for (let c = 0; c < 4; c++)
          for (let a = 0; a < 3; a++)
            f.pose.jointMatrices[offset + c * 4 + a] = (R() - 0.5) * (c === 3 ? 80 : 4);
      }
      f.pose.version++;
      const snapshot = b.update();
      for (const options of [
        {},
        { fused: true },
        { reverse: true },
        { flush: true, fused: true, reverse: true },
      ])
        for (const p of reference(f.pose, f.geometry, options)) {
          contains(snapshot, p);
          evaluated++;
        }
    }
  }
  assert.equal(evaluated, 19680);
});
test("subnormal input and output flushing stay enclosed", () => {
  const f = fixture({ targets: 1, joints: 1 });
  f.geometry.positions.fill(2 ** -140);
  f.geometry.morphTargets[0].positions.fill(2 ** -125);
  f.pose.morphWeights[0] = -0.25;
  f.pose.jointMatrices[16] = 0.1;
  f.pose.jointMatrices[21] = 0.1;
  f.pose.jointMatrices[26] = 0.1;
  const b = createAnimationBounds(f.pose, f.geometry);
  for (const options of [{}, { flush: true }, { fused: true }])
    for (const p of reference(f.pose, f.geometry, options)) contains(b.snapshot, p);
});
test("overflow becomes unbounded and cannot hide geometry even behind a zero transform", () => {
  const f = fixture({ targets: 1, joints: 1 });
  f.geometry.morphTargets[0].positions.fill(1e30);
  f.pose.morphWeights[0] = 1e30;
  f.pose.jointMatrices[16] = f.pose.jointMatrices[21] = f.pose.jointMatrices[26] = 0;
  const b = createAnimationBounds(f.pose, f.geometry);
  assert.equal(b.snapshot.bounded, false);
  const m = I();
  m[12] = 1e20;
  assert.equal(animationBoundsVisible(b.snapshot, I(), m), true);
});
test("bad pose data fails without replacing a prior snapshot and can recover", () => {
  const f = fixture({ targets: 1, joints: 1 }),
    b = createAnimationBounds(f.pose, f.geometry),
    before = b.snapshot;
  f.pose.morphWeights[0] = NaN;
  assert.throws(() => b.update(), { code: "ANIMATION_BOUNDS_VALUE" });
  assert.equal(b.snapshot, before);
  f.pose.morphWeights[0] = 0;
  f.pose.jointMatrices[19] = 1;
  assert.throws(() => b.update(), { code: "ANIMATION_BOUNDS_POSE" });
  f.pose.jointMatrices[19] = 0;
  assert.equal(b.update().bounded, true);
});
test("limits, illegal weights and storage are refused before use", () => {
  const f = fixture({ joints: 1, targets: 1 });
  assert.throws(() => createAnimationBounds(f.pose, f.geometry, { maxComponents: 1 }), {
    code: "ANIMATION_BOUNDS_LIMIT",
  });
  assert.throws(() => createAnimationBounds(f.pose, f.geometry, { maxBytes: 95 }), {
    code: "ANIMATION_BOUNDS_LIMIT",
  });
  const good = f.geometry.weights[0];
  f.geometry.weights[0] = -1;
  assert.throws(() => createAnimationBounds(f.pose, f.geometry));
  f.geometry.weights[0] = good;
  f.geometry.joints[0] = 999;
  assert.throws(() => createAnimationBounds(f.pose, f.geometry));
  f.geometry.joints[0] = 0;
  f.geometry.positions = new Float32Array(new SharedArrayBuffer(36));
  assert.throws(() => createAnimationBounds(f.pose, f.geometry), {
    code: "ANIMATION_BOUNDS_STORAGE",
  });
});
test("detached poses and disposal fail rather than returning stale bounds", () => {
  const f = fixture({ targets: 1 }),
    b = createAnimationBounds(f.pose, f.geometry);
  structuredClone(f.pose.morphWeights.buffer, { transfer: [f.pose.morphWeights.buffer] });
  assert.throws(() => b.update());
  b.dispose();
  assert.throws(() => b.update(), { code: "ANIMATION_BOUNDS_DISPOSED" });
  assert.throws(() => b.snapshot, { code: "ANIMATION_BOUNDS_DISPOSED" });
});
for (const [axis, side] of [
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 1],
  [2, -1],
  [2, 1],
])
  test(`clip rejection outside axis ${axis}, side ${side}`, () => {
    const min = [-0.1, -0.1, 0.4],
      max = [0.1, 0.1, 0.6];
    min[axis] = side < 0 ? -3 : 2;
    max[axis] = side < 0 ? -2 : 3;
    assert.equal(animationBoundsVisible(box(min, max), I(), I()), false);
  });
test("touching, straddling, enclosing and empty-direction perspective cases are retained", () => {
  for (const b of [
    box([-1, -1, 0], [1, 1, 1]),
    box([1, -0.1, 0.4], [2, 0.1, 0.6]),
    box([-100, -100, -100], [100, 100, 100]),
    box([0, 0, 0], [0, 0, 0]),
  ])
    assert.equal(animationBoundsVisible(b, I(), I()), true);
  assert.equal(
    animationBoundsVisible(box([-0.1, -0.1, 1], [0.1, 0.1, 2]), perspective(), I()),
    false,
  );
  assert.equal(animationBoundsVisible(box([-1, -1, -2], [1, 1, 2]), perspective(), I()), true);
});
test("world transforms, infinite far and reversed-depth projections are handled without inverses", () => {
  const b = box([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]),
    m = I();
  m[14] = -3;
  assert.equal(animationBoundsVisible(b, perspective(), m), true);
  m[12] = 100;
  assert.equal(animationBoundsVisible(b, perspective(), m), false);
  m[12] = 0;
  m[14] = -1e6;
  const inf = perspective();
  inf[10] = -1;
  inf[14] = -1;
  assert.equal(animationBoundsVisible(b, inf, m), true);
  const reverse = I();
  reverse[10] = -1;
  reverse[14] = 1;
  assert.equal(animationBoundsVisible(box([0, 0, 0.3], [0.1, 0.1, 0.7]), reverse, I()), true);
  assert.equal(animationBoundsVisible(box([0, 0, 2], [0.1, 0.1, 3]), reverse, I()), false);
});
test("culling must retain any box with a visible reference vertex over randomized view transforms", () => {
  let s = 888;
  const r = () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return s / 2 ** 32;
  };
  let visible = 0;
  for (let sample = 0; sample < 2000; sample++) {
    const points = Array.from({ length: 12 }, () => [
      (r() - 0.5) * 10,
      (r() - 0.5) * 10,
      (r() - 0.5) * 10,
    ]);
    const b = box(
      [0, 1, 2].map((a) => Math.min(...points.map((p) => p[a]))),
      [0, 1, 2].map((a) => Math.max(...points.map((p) => p[a]))),
    );
    const w = I();
    for (let c = 0; c < 4; c++)
      for (let a = 0; a < 3; a++) w[c * 4 + a] = (r() - 0.5) * (c === 3 ? 20 : 2);
    const vp = sample % 2 ? perspective() : I(),
      clip = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let a = 0; a < 4; a++)
        clip[c * 4 + a] =
          vp[a] * w[c * 4] +
          vp[a + 4] * w[c * 4 + 1] +
          vp[a + 8] * w[c * 4 + 2] +
          vp[a + 12] * w[c * 4 + 3];
    const hit = points.some((p) => {
      const q = [0, 1, 2, 3].map((a) =>
        Math.fround(clip[a] * p[0] + clip[a + 4] * p[1] + clip[a + 8] * p[2] + clip[a + 12]),
      );
      return (
        q[0] >= -q[3] && q[0] <= q[3] && q[1] >= -q[3] && q[1] <= q[3] && q[2] >= 0 && q[2] <= q[3]
      );
    });
    if (hit) {
      visible++;
      assert.equal(animationBoundsVisible(b, vp, w), true);
    }
  }
  assert.ok(visible > 50);
});
test("invalid transforms are errors while overflow/unknown bounds are fail-open", () => {
  const b = box([0, 0, 0], [1, 1, 1]),
    w = I();
  w[0] = NaN;
  assert.throws(() => animationBoundsVisible(b, I(), w));
  w[0] = 1e300;
  assert.equal(animationBoundsVisible(b, I(), w), true);
  assert.equal(animationBoundsVisible(box([NaN, 0, 0], [1, 1, 1]), I(), I()), true);
});
