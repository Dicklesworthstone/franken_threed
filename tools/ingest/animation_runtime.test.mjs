import assert from "node:assert/strict";
import test from "node:test";
import { AnimationPoseError, createAnimationPlayer } from "./animation_runtime.mjs";

const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const definition = (extra) => ({ format: "f3d-animation-v1", nodes: [{}], ...extra });
const close = (a, b, tolerance = 1e-11) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) <= tolerance, `${i}: ${a[i]} vs ${b[i]}`);
};
const channel = (path, times, values, more = {}) => ({ node: 0, path, times, values, ...more });
const track = (path, times, values, more = {}) =>
  definition({ clips: [{ channels: [channel(path, times, values, more)] }] });
function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}
const translation = (x, y, z) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

test("rest pose composes parents before children even with shuffled node order", () => {
  const player = createAnimationPlayer(
    definition({
      nodes: [
        { parent: 2, translation: [0, 4, 0] },
        { translation: [20, 0, 0] },
        { parent: 3, translation: [3, 0, 0] },
        { translation: [1, 2, 3] },
      ],
    }),
  );
  close(player.worldMatrices.subarray(0, 16), translation(4, 6, 3));
  close(player.worldMatrices.subarray(16, 32), translation(20, 0, 0));
  assert.equal(player.version, 0);
  assert.equal(player.clip, -1);
  assert.equal(player.nodeCount, 4);
  const world = player.worldMatrices;
  player.reset();
  assert.equal(player.worldMatrices, world);
  assert.equal(player.version, 1);
});

test("deep hierarchy avoids recursion and reset ignores edits to published arrays", () => {
  const count = 10000,
    nodes = Array.from({ length: count }, (_, i) => ({
      parent: i === count - 1 ? -1 : i + 1,
      translation: [0.001, 0, 0],
    }));
  const player = createAnimationPlayer(definition({ nodes }));
  assert.ok(Math.abs(player.worldMatrices[12] - 10) < 1e-9);
  player.translations.fill(42);
  player.worldMatrices.fill(7);
  player.reset();
  assert.ok(Math.abs(player.worldMatrices[12] - 10) < 1e-9);
});

test("scale and quaternion composition matches independent matrix products", () => {
  const h = Math.SQRT1_2;
  const player = createAnimationPlayer(
    definition({ nodes: [{ translation: [1, 2, 3], rotation: [0, 0, h, h], scale: [2, 3, 4] }] }),
  );
  const R = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    S = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];
  close(player.worldMatrices, multiply(multiply(translation(1, 2, 3), R), S));
});

test("matrix nodes, negative scale, multiple roots and explicit root matrix", () => {
  const player = createAnimationPlayer(
    definition({ nodes: [{ matrix: translation(5, 6, 7) }, { parent: 0, scale: [-2, 1, 1] }] }),
  );
  player.sample(0, { clip: -1, rootMatrix: translation(10, 20, 30) });
  close(player.worldMatrices.subarray(0, 16), translation(15, 26, 37));
  assert.equal(player.worldMatrices[16], -2);
});

for (const interpolation of ["LINEAR", "STEP"])
  test(`${interpolation} handles distinct key ranges, endpoints, reverse seeks and exact keys`, () => {
    const player = createAnimationPlayer(
      track("translation", [2, 4, 10], [2, 4, 6, 4, 8, 12, 10, 20, 30], { interpolation }),
    );
    const before = player.translations;
    for (const t of [-100, 0, 2, 3, 4, 7, 10, 20, 3.5, 2.1, 9.9, 4, 2, 0]) {
      player.sample(t);
      const x =
        interpolation === "STEP" ? (t < 4 ? 2 : t < 10 ? 4 : 10) : Math.max(2, Math.min(10, t));
      close(player.translations, [x, x * 2, x * 3]);
      assert.equal(player.translations, before);
    }
  });

test("single-key clips, clip switching and explicit negative loop time reset rest properties", () => {
  const d = definition({
    nodes: [{ translation: [7, 8, 9], scale: [2, 3, 4] }],
    clips: [
      { name: "move", channels: [channel("translation", [2], [10, 20, 30])] },
      { name: "scale", channels: [channel("scale", [0, 4], [1, 1, 1, 5, 5, 5])] },
    ],
  });
  const p = createAnimationPlayer(d);
  p.sample(99);
  close(p.translations, [10, 20, 30]);
  close(p.scales, [2, 3, 4]);
  p.sample(-1, { clip: 1, loop: true });
  assert.equal(p.time, 3);
  close(p.scales, [4, 4, 4]);
  close(p.translations, [7, 8, 9]);
  p.sample(4, { clip: 1, loop: true });
  close(p.scales, [1, 1, 1]);
  p.reset();
  close(p.translations, [7, 8, 9]);
  close(p.scales, [2, 3, 4]);
  d.nodes[0].translation[0] = 999;
  d.clips[0].channels[0].values[0] = 888;
  p.sample(0);
  assert.equal(p.translations[0], 10);
});

test("CUBICSPLINE scales derivatives by the actual interval duration", () => {
  // x(t)=t^3, keys t=1 and 3 with derivatives 3t^2.
  const values = [0, 0, 0, 1, 2, 3, 3, 6, 9, 27, 54, 81, 27, 54, 81, 0, 0, 0];
  const p = createAnimationPlayer(
    track("translation", [1, 3], values, { interpolation: "CUBICSPLINE" }),
  );
  for (const time of [0, 1, 1.25, 2, 2.8, 3, 4]) {
    p.sample(time);
    const x = Math.max(1, Math.min(3, time)) ** 3;
    close(p.translations, [x, x * 2, x * 3]);
  }
});

test("quaternion LINEAR uses shortest-path SLERP, with correct endpoints and signed equivalence", () => {
  const p = createAnimationPlayer(track("rotation", [0, 2], [0, 0, 0, 1, 0, 0, 1, 0]));
  p.sample(1);
  close(p.rotations, [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  close(p.worldMatrices, multiply(I, [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 1e-12);
  const opposite = createAnimationPlayer(track("rotation", [0, 2], [0, 0, 0, 1, 0, 0, 0, -1]));
  opposite.sample(1);
  close(opposite.rotations, [0, 0, 0, 1]);
  opposite.sample(2);
  close(opposite.rotations, [0, 0, 0, -1]);
});

test("cubic rotations normalize the Hermite result, without negating tangents", () => {
  const p = createAnimationPlayer(
    track(
      "rotation",
      [0, 2],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      { interpolation: "CUBICSPLINE" },
    ),
  );
  p.sample(1);
  const length = Math.hypot(0.75, 0.5);
  close(p.rotations, [0, 0, 0.75 / length, 0.5 / length]);
  assert.ok(Math.abs(Math.hypot(...p.rotations) - 1) < 1e-15);
});

for (const interpolation of ["LINEAR", "STEP", "CUBICSPLINE"])
  test(`morph weight streams retain channel width and node offsets: ${interpolation}`, () => {
    const data =
      interpolation === "CUBICSPLINE"
        ? [0, 0, 0.1, 0.9, 0, 0, 0, 0, 0.9, 0.1, 0, 0]
        : [0.1, 0.9, 0.9, 0.1];
    const p = createAnimationPlayer(
      definition({
        nodes: [{ weights: [0, 0] }, { weights: [0.3] }],
        clips: [{ channels: [channel("weights", [0, 2], data, { interpolation })] }],
      }),
    );
    p.sample(1);
    close(p.morphWeights, interpolation === "STEP" ? [0.1, 0.9, 0.3] : [0.5, 0.5, 0.3]);
    assert.deepEqual([...p.morphOffsets], [0, 2, 3]);
  });

test("joint ordering, non-joint ancestors, inverse bind and mesh-local palettes", () => {
  const p = createAnimationPlayer(
    definition({
      nodes: [
        { translation: [5, 0, 0] }, // mesh
        { translation: [10, 0, 0] }, // non-joint ancestor
        { parent: 1, translation: [0, 2, 0] },
        { parent: 2, translation: [0, 0, 3] },
        { translation: [-5, 0, 0] }, // same skin on another mesh
      ],
      skins: [
        {
          joints: [3, 2],
          inverseBindMatrices: [...translation(-10, -2, -3), ...translation(-10, -2, 0)],
        },
      ],
      instances: [
        { node: 0, skin: 0 },
        { node: 4, skin: 0 },
      ],
    }),
  );
  assert.deepEqual(
    p.instances.map((x) => x.offset),
    [0, 32],
  );
  close(p.jointMatrices.subarray(0, 16), translation(-5, 0, 0));
  close(p.jointMatrices.subarray(16, 32), translation(-5, 0, 0));
  close(p.jointMatrices.subarray(32, 48), translation(5, 0, 0));
  p.sample(0, { clip: -1, rootMatrix: translation(100, 200, 300) });
  close(p.jointMatrices.subarray(0, 16), translation(-5, 0, 0));
});

test("general affine inverse handles hierarchy-induced shear and negative scales", () => {
  const q = [0, 0, Math.sin(0.3), Math.cos(0.3)];
  const p = createAnimationPlayer(
    definition({
      nodes: [
        { scale: [2, -3, 4] },
        { parent: 0, rotation: q, translation: [7, 8, 9] },
        { translation: [3, 4, 5] },
      ],
      skins: [{ joints: [2] }],
      instances: [{ node: 1, skin: 0 }],
    }),
  );
  const reconstructed = multiply([...p.worldMatrices.subarray(16, 32)], [...p.jointMatrices]);
  close(reconstructed, [...p.worldMatrices.subarray(32, 48)], 2e-6);
});

test("sampling joint animation drives a real weighted vertex result", () => {
  const p = createAnimationPlayer(
    definition({
      nodes: [{}, {}],
      skins: [{ joints: [1] }],
      instances: [{ node: 0, skin: 0 }],
      clips: [{ channels: [channel("translation", [0, 2], [0, 0, 0, 2, 4, 6], { node: 1 })] }],
    }),
  );
  for (const time of [0, 0.5, 1, 2, 1, 0]) {
    p.sample(time);
    const m = p.jointMatrices,
      x = 1,
      y = 2,
      z = 3;
    close(
      [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
      ],
      [1 + time, 2 + 2 * time, 3 + 3 * time],
    );
  }
});

test("bad times, zero interpolated quaternion and singular mesh publish nothing", () => {
  const p = createAnimationPlayer(
    definition({
      nodes: [{}, {}],
      skins: [{ joints: [1] }],
      instances: [{ node: 0, skin: 0 }],
      clips: [{ channels: [channel("scale", [0, 1], [1, 1, 1, 0, 0, 0])] }],
    }),
  );
  p.sample(0.5);
  const snapshot = Object.fromEntries(
      ["translations", "rotations", "scales", "worldMatrices", "jointMatrices"].map((k) => [
        k,
        p[k].slice(),
      ]),
    ),
    version = p.version;
  for (const fn of [
    () => p.sample(NaN),
    () => p.sample(Infinity),
    () => p.sample(1),
    () => p.sample(0, { clip: 99 }),
    () => p.sample(0, { loop: "true" }),
  ])
    assert.throws(fn, AnimationPoseError);
  for (const [k, values] of Object.entries(snapshot)) assert.deepEqual(p[k], values);
  assert.equal(p.version, version);
  p.sample(0.1);
  assert.equal(p.version, version + 1);
  const bad = createAnimationPlayer(
    track(
      "rotation",
      [0, 2],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
      { interpolation: "CUBICSPLINE" },
    ),
  );
  assert.throws(() => bad.sample(1), { code: "ANIMATION_QUATERNION" });
  close(bad.rotations, [0, 0, 0, 1]);
  assert.equal(bad.version, 0);
});

test("instances are independent and disposal prevents further updates", () => {
  const data = track("translation", [0, 1], [0, 0, 0, 1, 2, 3]),
    a = createAnimationPlayer(data),
    b = createAnimationPlayer(data);
  a.sample(0.5);
  close(b.translations, [0, 0, 0]);
  assert.notEqual(a.worldMatrices, b.worldMatrices);
  a.dispose();
  assert.equal(a.disposed, true);
  assert.throws(() => a.reset(), { code: "ANIMATION_DISPOSED" });
  b.sample(1);
  close(b.translations, [1, 2, 3]);
});

const invalid = [
  definition({ nodes: [{ parent: 1 }, { parent: 0 }] }),
  definition({ nodes: [{ parent: 0 }] }),
  definition({ nodes: [{ parent: 999 }] }),
  definition({ nodes: [{ rotation: [0, 0, 0, 0] }] }),
  definition({ nodes: [{ matrix: I, translation: [0, 0, 0] }] }),
  definition({ skins: [{ joints: [0, 0] }] }),
  definition({ skins: [{ joints: [1] }] }),
  definition({ skins: [{ joints: [0], inverseBindMatrices: [] }] }),
  definition({
    skins: [{ joints: [0] }],
    instances: [
      { node: 0, skin: 0 },
      { node: 0, skin: 0 },
    ],
  }),
  track("translation", [0, 0], [0, 0, 0, 1, 1, 1]),
  track("translation", [-1, 1], [0, 0, 0, 1, 1, 1]),
  track("translation", [0, 1], [0, 0, 0]),
  track("rotation", [0], [0, 0, 0, 0]),
  track("weights", [0], [1]),
  track("translation", [0], [0, 0, 0], { interpolation: "CUBICSPLINE" }),
  track("translation", [0], [0, 0, 0], { interpolation: "UNKNOWN" }),
  track("visibility", [0], [1]),
  definition({
    clips: [
      {
        channels: [channel("translation", [0], [1, 2, 3]), channel("translation", [0], [3, 2, 1])],
      },
    ],
  }),
  { ...track("translation", [0], [1, 2, 3]), nodes: [{ matrix: I }] },
];
for (let i = 0; i < invalid.length; i++)
  test(`invalid definition ${i} is refused at construction`, () =>
    assert.throws(() => createAnimationPlayer(invalid[i]), AnimationPoseError));

test("detached published storage is rejected before any other pose field is changed", () => {
  const p = createAnimationPlayer(track("translation", [0, 1], [0, 0, 0, 1, 2, 3]));
  p.sample(0.25);
  const before = p.translations.slice(),
    version = p.version;
  structuredClone(p.jointMatrices.buffer, { transfer: [p.jointMatrices.buffer] });
  assert.throws(() => p.sample(0.75), { code: "ANIMATION_OUTPUT_STORAGE" });
  assert.deepEqual(p.translations, before);
  assert.equal(p.version, version);
});

test("positive times already inside the loop interval do not acquire modulo rounding error", () => {
  const p = createAnimationPlayer(track("translation", [0, 2], [0, 0, 0, 2, 4, 6]));
  for (const time of [Number.MIN_VALUE, 1e-20, 0.1, 0.3, 1.9]) {
    p.sample(time, { loop: true });
    assert.equal(p.time, time);
  }
});
