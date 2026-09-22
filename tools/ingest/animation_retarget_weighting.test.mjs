import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationRetargeter } from "./animation_retarget.mjs";
import { retargetAnimationClip } from "./animation_retarget_clip.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

const q = (a) => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
const definition = () => ({
  format: "f3d-animation-v1",
  nodes: [{}, { parent: 0, translation: [1, 0, 0] }, { parent: 1, translation: [1, 0, 0] }],
});
const mapping = [
  { source: 0, target: 0 },
  { source: 1, target: 1 },
];
function rotation(p, node, angle) {
  const expected = q(angle),
    actual = p.snapshotLocalPose().rotations.slice(node * 4, node * 4 + 4);
  actual.forEach((v, i) =>
    assert.ok(Math.abs(v - expected[i]) < 1e-9, `${node}: ${v} != ${expected[i]}`),
  );
}
// The source rotates its parent and child locally by 1 and .4 radians. A local
// partial overlay must not add extra child motion to compensate a blended parent.
for (const weight of [0.25, 0.5])
  test(`partial skeletal weight ${weight} does not amplify inherited parent rotation`, () => {
    const s = createAnimationPlayer(definition()),
      t = createAnimationPlayer(definition()),
      r = createAnimationRetargeter(s, t, { mapping });
    s.edit([
      { node: 0, rotation: q(1) },
      { node: 1, rotation: q(0.4) },
    ]);
    r.apply({ weight });
    rotation(t, 0, weight);
    rotation(t, 1, 0.4 * weight);
    r.dispose();
    s.dispose();
    t.dispose();
  });
test("per-bone overlay weights do not introduce a compensating child twist", () => {
  const s = createAnimationPlayer(definition()),
    t = createAnimationPlayer(definition()),
    r = createAnimationRetargeter(s, t, {
      mapping: [
        { source: 0, target: 0, weight: 0.5 },
        { source: 1, target: 1 },
      ],
    });
  s.edit([{ node: 0, rotation: q(1) }]);
  r.apply();
  rotation(t, 0, 0.5);
  rotation(t, 1, 0);
  r.dispose();
  s.dispose();
  t.dispose();
});
test("baked partial skeleton mappings do not accumulate parent motion in children", () => {
  const source = definition(),
    target = definition();
  source.clips = [
    {
      channels: [
        { node: 0, path: "rotation", times: [0, 1], values: [...q(0), ...q(Math.PI / 2)] },
      ],
    },
  ];
  const baked = retargetAnimationClip(source, target, {
      mapping: mapping.map((b) => ({ ...b, weight: 0.5 })),
      frameRate: 4,
    }),
    p = createAnimationPlayer({ ...target, clips: [baked.clip] });
  p.sample(1);
  rotation(p, 0, Math.PI / 4);
  rotation(p, 1, 0);
  p.dispose();
});
