import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationRetargeter } from "./animation_retarget.mjs";
import { mapAnimationNodeNames, retargetAnimationClip } from "./animation_retarget_clip.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const q = (a) => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
const def = (nodes, extra = {}) => ({ format: "f3d-animation-v1", nodes, ...extra });
const close = (a, b, e = 1e-8) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < e, `${i}: ${v} != ${b[i]}`));
};
function fixture(interpolation = "LINEAR") {
  const channels = [
    {
      node: 0,
      path: "rotation",
      times: [0, 1],
      values: [...q(0), ...q(Math.PI / 2)],
      interpolation,
    },
    { node: 0, path: "translation", times: [0, 1], values: [0, 0, 0, 4, 0, 0], interpolation },
  ];
  return {
    source: def(
      [
        { name: "Hip" },
        { parent: 0, name: "Arm", translation: [1, 0, 0] },
        { parent: 1, translation: [1, 0, 0] },
      ],
      { clips: [{ name: "walk", channels }] },
    ),
    target: def([
      { name: "Pelvis" },
      { parent: 0, name: "Hand", translation: [2, 0, 0] },
      { parent: 1, translation: [2, 0, 0] },
    ]),
    options: {
      mapping: [
        { source: 0, target: 0 },
        { source: 1, target: 1 },
      ],
      rootMotion: { source: 0, target: 0, scale: 2 },
      frameRate: 4,
    },
  };
}
const playback = (target, baked) => createAnimationPlayer({ ...target, clips: [baked.clip] });

test("bakes reusable JSON-serializable local tracks with scaled root motion and destination proportions", () => {
  const f = fixture(),
    before = structuredClone(f),
    r = retargetAnimationClip(f.source, f.target, f.options);
  assert.equal(r.frameCount, 5);
  assert.equal(r.clip.name, "walk_retargeted");
  assert.equal(r.duration, 1);
  assert.equal(r.approximate, true);
  assert.equal(r.components, 5 * (5 + 5 + 4));
  assert.deepEqual(r.sourceRange, [0, 1]);
  assert.deepEqual(f, before);
  const copy = JSON.parse(JSON.stringify(r.clip)),
    p = createAnimationPlayer({ ...f.target, clips: [copy] });
  p.sample(0.5);
  close(p.translations.slice(0, 3), [4, 0, 0]);
  close(p.rotations.slice(0, 4), q(Math.PI / 4));
  close(p.worldMatrices.slice(44, 47), [4 + 2 * Math.SQRT2, 2 * Math.SQRT2, 0]);
  assert.deepEqual(
    r.clip.channels.map((c) => [c.node, c.path]),
    [
      [0, "rotation"],
      [1, "rotation"],
      [0, "translation"],
    ],
  );
  for (const c of r.clip.channels) assert.deepEqual(c.times, [0, 0.25, 0.5, 0.75, 1]);
  assert.notEqual(r.clip.channels[0].times, r.clip.channels[1].times);
  p.dispose();
});

test("every sample matches direct live retargeting with the same imported reference stances", () => {
  const f = fixture();
  f.target.nodes[1].rotation = q(-0.4);
  f.options.alignment = q(0.3);
  const r = retargetAnimationClip(f.source, f.target, { ...f.options, frameRate: 10 }),
    s = createAnimationPlayer(f.source),
    t = createAnimationPlayer(f.target),
    live = createAnimationRetargeter(s, t, {
      mapping: f.options.mapping,
      rootMotion: f.options.rootMotion,
      alignment: f.options.alignment,
    }),
    p = playback(f.target, r);
  for (const time of r.clip.channels[0].times) {
    s.sample(time);
    t.reset();
    live.apply();
    p.sample(time);
    close(p.worldMatrices, t.worldMatrices);
  }
  p.dispose();
  s.dispose();
  t.dispose();
  live.dispose();
});

test("partial mapping weights are reset to target rest before every frame, not accumulated", () => {
  const f = fixture();
  f.options.mapping = [{ source: 0, target: 0, weight: 0.5 }];
  const r = retargetAnimationClip(f.source, f.target, f.options),
    p = playback(f.target, r);
  p.sample(1);
  close(p.rotations.slice(0, 4), q(Math.PI / 4));
  p.dispose();
});

test("sampling uses an explicitly selected clip and does not include other source motion", () => {
  const f = fixture();
  f.source.clips.push({
    name: "other",
    channels: [{ node: 0, path: "rotation", times: [0, 2], values: [...q(0), ...q(-1)] }],
  });
  const r = retargetAnimationClip(f.source, f.target, {
      ...f.options,
      clip: 1,
      name: "destination turn",
    }),
    p = playback(f.target, r);
  assert.equal(r.duration, 2);
  assert.equal(r.clip.name, "destination turn");
  p.sample(2);
  close(p.rotations.slice(0, 4), q(-1));
  close(p.translations.slice(0, 3), [0, 0, 0]);
  p.dispose();
});

test("trimmed ranges rebase output time to zero and include an off-grid endpoint once", () => {
  const f = fixture(),
    r = retargetAnimationClip(f.source, f.target, {
      ...f.options,
      start: 0.2,
      end: 0.85,
      frameRate: 4,
    });
  close(r.clip.channels[0].times, [0, 0.25, 0.5, 0.65]);
  assert.equal(r.frameCount, 4);
  const p = playback(f.target, r);
  p.sample(0);
  close(p.translations.slice(0, 3), [1.6, 0, 0]);
  p.sample(0.65);
  close(p.translations.slice(0, 3), [6.8, 0, 0]);
  p.dispose();
});

test("decimal range boundary does not add a duplicate endpoint frame", () => {
  const f = fixture(),
    r = retargetAnimationClip(f.source, f.target, {
      ...f.options,
      start: 0.1,
      end: 0.8,
      frameRate: 10,
      maxFrames: 8,
    });
  assert.equal(r.frameCount, 8);
  assert.equal(r.clip.channels[0].times.at(-1), 0.8 - 0.1);
});

test("zero-duration trim and constant-duration source clips produce one reusable sample", () => {
  const f = fixture(),
    r = retargetAnimationClip(f.source, f.target, {
      ...f.options,
      start: 0.5,
      end: 0.5,
      maxFrames: 1,
    }),
    p = playback(f.target, r);
  assert.equal(r.frameCount, 1);
  assert.equal(r.duration, 0);
  assert.deepEqual(r.clip.channels[0].times, [0]);
  p.sample(100);
  close(p.translations.slice(0, 3), [4, 0, 0]);
  p.dispose();
  f.source.clips = [{ channels: [{ node: 0, path: "rotation", times: [0], values: q(0.7) }] }];
  const constant = retargetAnimationClip(f.source, f.target, f.options);
  assert.equal(constant.frameCount, 1);
});

for (const interpolation of ["LINEAR", "STEP"])
  test(`bakes source STEP samples into explicitly selected ${interpolation} output`, () => {
    const f = fixture("STEP"),
      r = retargetAnimationClip(f.source, f.target, { ...f.options, interpolation }),
      p = playback(f.target, r);
    assert.ok(r.clip.channels.every((c) => c.interpolation === interpolation));
    p.sample(0.5);
    close(p.rotations.slice(0, 4), q(0));
    p.sample(1);
    close(p.rotations.slice(0, 4), q(Math.PI / 2));
    p.dispose();
  });

test("source cubic quaternion/translation tracks use the real Hermite sampler, not linear key copying", () => {
  const f = fixture();
  f.source.clips[0].channels = [
    {
      node: 0,
      path: "rotation",
      times: [0, 1],
      interpolation: "CUBICSPLINE",
      values: [0, 0, 0, 0, ...q(0), 0, 0, 1, 0, 0, 0, -1, 0, ...q(1), 0, 0, 0, 0],
    },
    {
      node: 0,
      path: "translation",
      times: [0, 1],
      interpolation: "CUBICSPLINE",
      values: [0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    },
  ];
  const r = retargetAnimationClip(f.source, f.target, f.options),
    s = createAnimationPlayer(f.source),
    p = playback(f.target, r);
  s.sample(0.5);
  p.sample(0.5);
  close(p.rotations, s.rotations);
  close(
    p.translations.slice(0, 3),
    Array.from(s.translations.slice(0, 3), (v) => v * 2),
  );
  assert.notEqual(p.translations[0], 4);
  p.dispose();
  s.dispose();
});

test("quaternion sign continuity preserves antipodal source orientations without huge flips", () => {
  const f = fixture("STEP");
  f.source.clips[0].channels[0] = {
    node: 0,
    path: "rotation",
    times: [0, 0.5, 1],
    interpolation: "STEP",
    values: [...q(0.3), ...q(0.3).map((v) => -v), ...q(0.3)],
  };
  const r = retargetAnimationClip(f.source, f.target, f.options),
    values = r.clip.channels[0].values;
  for (let i = 4; i < values.length; i += 4)
    assert.ok(values.slice(i, i + 4).reduce((s, v, k) => s + v * values[i - 4 + k], 0) > 0.999999);
});

test("baked skeleton clips update all mesh-local palettes and preserve target morph defaults", () => {
  const f = fixture();
  f.target.nodes.push(
    { translation: [10, 0, 0], weights: [0.3, 0.7] },
    { translation: [20, 0, 0] },
  );
  f.target.skins = [{ joints: [0, 1] }];
  f.target.instances = [
    { node: 3, skin: 0 },
    { node: 4, skin: 0 },
  ];
  const r = retargetAnimationClip(f.source, f.target, f.options),
    p = playback(f.target, r);
  p.sample(0.5);
  assert.equal(p.jointMatrices[12], -6);
  assert.equal(p.jointMatrices[44], -16);
  close(p.morphWeights, [0.3, 0.7]);
  p.dispose();
});

test("translation-only bindings and zero-weight rotational bindings emit only the active tracks", () => {
  const f = fixture();
  f.options.mapping = [{ source: 0, target: 0, weight: 0 }];
  const r = retargetAnimationClip(f.source, f.target, f.options);
  assert.deepEqual(
    r.clip.channels.map((c) => c.path),
    ["translation"],
  );
  assert.equal(r.components, 20);
});

for (const options of [
  { frameRate: 0 },
  { frameRate: 1001 },
  { frameRate: NaN },
  { start: -1 },
  { end: 2 },
  { start: 0.8, end: 0.2 },
  { clip: 99 },
  { interpolation: "CUBICSPLINE" },
  { maxFrames: 4 },
  { maxComponents: 69 },
  { maxWork: 29 },
  { maxNodes: 5 },
  { maxMappings: 1 },
  { maxFrames: 0 },
  { name: 42 },
  { unknown: true },
])
  test(
    "invalid/over-budget baking fails without changing definitions " + JSON.stringify(options),
    () => {
      const f = fixture(),
        before = structuredClone(f);
      assert.throws(() => retargetAnimationClip(f.source, f.target, { ...f.options, ...options }));
      assert.deepEqual(f, before);
    },
  );

test("exact output, frame and node-frame budgets succeed and are independently enforced", () => {
  const f = fixture(),
    r = retargetAnimationClip(f.source, f.target, {
      ...f.options,
      maxFrames: 5,
      maxComponents: 70,
      maxWork: 30,
      maxNodes: 6,
      maxMappings: 2,
    });
  assert.equal(r.frameCount, 5);
  assert.equal(r.components, 70);
});

test("an invalid later frame publishes no partial clip and does not mutate input definitions", () => {
  const f = fixture();
  f.source.clips[0].channels.push({
    node: 0,
    path: "scale",
    times: [0, 1],
    values: [1, 1, 1, 2, 1, 1],
  });
  const before = structuredClone(f);
  assert.throws(() => retargetAnimationClip(f.source, f.target, f.options), {
    code: "ANIMATION_RETARGET_TRANSFORM",
  });
  assert.deepEqual(f, before);
});

test("abort before or between sample iterations cannot expose a partial clip", () => {
  const f = fixture(),
    c = new AbortController(),
    reason = Error("cancel");
  c.abort(reason);
  assert.throws(
    () => retargetAnimationClip(f.source, f.target, { ...f.options, signal: c.signal }),
    (e) => e === reason,
  );
  // A cooperative host signal is read at each bounded synchronous iteration.
  let polls = 0;
  const signal = {
    get aborted() {
      return ++polls > 4;
    },
    reason,
  };
  assert.throws(
    () => retargetAnimationClip(f.source, f.target, { ...f.options, signal }),
    (e) => e === reason,
  );
});

test("input float time spacing below numeric resolution fails explicitly instead of duplicate keys", () => {
  const f = fixture();
  f.source.clips[0].channels = [
    { node: 0, path: "translation", times: [0, 1e16 + 4], values: [0, 0, 0, 1, 0, 0] },
  ];
  assert.throws(
    () => retargetAnimationClip(f.source, f.target, { ...f.options, start: 1e16, end: 1e16 + 4 }),
    { code: "ANIMATION_RETARGET_RANGE" },
  );
});

test("exact named aliases produce frozen bindings usable for live and baked motion", () => {
  const f = fixture(),
    names = [
      { source: "Hip", target: "Pelvis" },
      { source: "Arm", target: "Hand", weight: 0.5 },
    ],
    map = mapAnimationNodeNames(f.source, f.target, names);
  assert.deepEqual(map, [
    { source: 0, target: 0, weight: 1 },
    { source: 1, target: 1, weight: 0.5 },
  ]);
  assert.ok(Object.isFrozen(map) && Object.isFrozen(map[0]));
  names[0].source = "missing";
  const r = retargetAnimationClip(f.source, f.target, { ...f.options, mapping: map });
  assert.equal(r.frameCount, 5);
});

for (const pairs of [
  [{ source: "Missing", target: "Pelvis" }],
  [{ source: "Hip", target: "Missing" }],
  [{ source: "hip", target: "Pelvis" }],
  [
    { source: "Hip", target: "Pelvis" },
    { source: "Arm", target: "Pelvis" },
  ],
  [{ source: "Hip", target: "Pelvis", weight: 2 }],
  [],
  [{ source: "Hip", target: "Pelvis", unknown: 1 }],
])
  test("named mappings reject missing/ambiguous intent " + JSON.stringify(pairs), () => {
    const f = fixture();
    assert.throws(() => mapAnimationNodeNames(f.source, f.target, pairs));
  });

test("selected duplicate names fail rather than selecting the first bone", () => {
  const f = fixture();
  f.source.nodes[2].name = "Hip";
  assert.throws(
    () => mapAnimationNodeNames(f.source, f.target, [{ source: "Hip", target: "Pelvis" }]),
    { code: "ANIMATION_RETARGET_MAPPING" },
  );
  f.source.nodes[2].name = "Arm";
  assert.deepEqual(
    mapAnimationNodeNames(f.source, f.target, [{ source: "Hip", target: "Pelvis" }]),
    [{ source: 0, target: 0, weight: 1 }],
  );
});

test("special object property names remain ordinary node-name data", () => {
  const s = def([{ name: "__proto__" }, { name: "constructor" }]),
    t = def([{ name: "prototype" }, { name: "toString" }]);
  assert.deepEqual(mapAnimationNodeNames(s, t, [{ source: "__proto__", target: "toString" }]), [
    { source: 0, target: 1, weight: 1 },
  ]);
});
