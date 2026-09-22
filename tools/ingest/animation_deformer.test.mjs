import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationDeformer } from "./animation_deformer.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

const near = (actual, expected, tolerance = 1e-6) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) =>
    assert.ok(
      Math.abs(value - expected[i]) <= tolerance,
      `component ${i}: ${value} != ${expected[i]}`,
    ),
  );
};
const code = (expected) => (error) => error.code === expected;
function animated() {
  return createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [
      { translation: [5, 0, 0], weights: [0] },
      { translation: [2, 0, 0], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] },
      { translation: [-7, 0, 0], weights: [0] },
    ],
    skins: [{ joints: [1] }],
    instances: [
      { node: 0, skin: 0 },
      { node: 2, skin: 0 },
    ],
    clips: [
      {
        channels: [
          { node: 1, path: "translation", times: [0, 2], values: [2, 0, 0, 4, 0, 0] },
          { node: 0, path: "weights", times: [0, 2], values: [0, 1] },
          { node: 2, path: "weights", times: [0, 2], values: [0, 1] },
        ],
      },
    ],
  });
}
function geometry(node = 0) {
  return {
    node,
    positions: [1, 0, 0],
    normals: [1, 0, 0],
    tangents: [1, 0, 0, -1],
    morphTargets: [{ positions: [2, 0, 0], normals: [0, 1, 0], tangents: [0, 2, 0] }],
    joints: [0, 0, 0, 0],
    weights: [1, 0, 0, 0],
  };
}

test("animated morphs precede skinning; normals/tangents exclude translation", () => {
  const pose = animated(),
    mesh = createAnimationDeformer(pose, geometry());
  assert.equal(mesh.version, 0);
  assert.equal(mesh.poseVersion, 0);
  near(mesh.positions, [-3, 1, 0]);
  const arrays = [
    mesh.positions,
    mesh.normals,
    mesh.tangents,
    mesh.worldMatrix,
    mesh.bounds.min,
    mesh.bounds.max,
  ];
  pose.sample(1);
  assert.equal(mesh.update(), mesh);
  near(mesh.positions, [-2, 2, 0]);
  near(mesh.normals, [-0.5, 1, 0]);
  near(mesh.tangents, [-1, 1, 0, -1]);
  near(mesh.bounds.min, mesh.positions);
  near(mesh.bounds.max, mesh.positions);
  assert.equal(mesh.worldMatrix[12], 5);
  assert.equal(mesh.poseVersion, pose.version);
  assert.equal(mesh.version, 1);
  assert.deepEqual(
    [
      mesh.positions,
      mesh.normals,
      mesh.tangents,
      mesh.worldMatrix,
      mesh.bounds.min,
      mesh.bounds.max,
    ],
    arrays,
  );
  for (let frame = 0; frame < 60; frame++) {
    pose.sample(frame / 30);
    mesh.update();
  }
  arrays.forEach((array, i) =>
    assert.equal(
      array,
      [
        mesh.positions,
        mesh.normals,
        mesh.tangents,
        mesh.worldMatrix,
        mesh.bounds.min,
        mesh.bounds.max,
      ][i],
    ),
  );
});

test("two nodes sharing a skin use separate mesh-local palettes", () => {
  const pose = animated(),
    a = createAnimationDeformer(pose, geometry()),
    b = createAnimationDeformer(pose, geometry(2));
  assert.notEqual(pose.instances[0].offset, pose.instances[1].offset);
  pose.sample(1);
  a.update();
  b.update();
  near(a.positions, [-2, 2, 0]);
  near(b.positions, [10, 2, 0]);
  near([a.positions[0] + a.worldMatrix[12], a.positions[1]], [3, 2]);
  near([b.positions[0] + b.worldMatrix[12], b.positions[1]], [3, 2]);
});

test("eight influences per vertex are retained, not truncated to four", () => {
  const nodes = [
    { translation: [3, 0, 0] },
    ...Array.from({ length: 8 }, (_, i) => ({ translation: [i, i * 2, 0] })),
  ];
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes,
    skins: [{ joints: [1, 2, 3, 4, 5, 6, 7, 8] }],
    instances: [{ node: 0, skin: 0 }],
  });
  const positions = [],
    joints = [],
    weights = [];
  for (let vertex = 0; vertex < 100; vertex++) {
    positions.push(vertex / 10, -vertex, 2);
    for (let joint = 0; joint < 8; joint++) {
      joints.push(joint);
      weights.push(0.125);
    }
  }
  const mesh = createAnimationDeformer(pose, {
    node: 0,
    positions,
    joints,
    weights,
    influences: 8,
  });
  for (let vertex = 0; vertex < 100; vertex++)
    near(mesh.positions.subarray(vertex * 3, vertex * 3 + 3), [vertex / 10 + 0.5, -vertex + 7, 2]);
  near(mesh.bounds.min, [0.5, -92, 2]);
  near(mesh.bounds.max, [10.4, 7, 2]);
});

test("morph-only meshes preserve signed and greater-than-one weights and omitted deltas", () => {
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [{ weights: [-1, 2], translation: [10, 0, 0] }],
  });
  const mesh = createAnimationDeformer(pose, {
    node: 0,
    positions: [1, 2, 3, -1, 0, 2],
    normals: [0, 1, 0, 0, 1, 0],
    morphTargets: [{ positions: [1, 0, 0, 0, 1, 0] }, { normals: [0, 0, 1, 0, 0, 1] }],
  });
  near(mesh.positions, [0, 2, 3, -1, -1, 2]);
  near(mesh.normals, [0, 1, 2, 0, 1, 2]);
  assert.equal(mesh.tangents, null);
  assert.equal(mesh.worldMatrix[12], 10);
  near(mesh.bounds.min, [-1, -1, 2]);
  near(mesh.bounds.max, [0, 2, 3]);
});

test("source geometry is snapshotted; updates repair caller-written outputs", () => {
  const pose = animated(),
    source = geometry(),
    mesh = createAnimationDeformer(pose, source);
  source.positions[0] = 100;
  source.morphTargets[0].positions[0] = 100;
  source.weights[0] = 0;
  mesh.positions.fill(900);
  mesh.normals.fill(900);
  mesh.bounds.min.fill(900);
  pose.sample(1);
  mesh.update();
  near(mesh.positions, [-2, 2, 0]);
  near(mesh.normals, [-0.5, 1, 0]);
  near(mesh.bounds.min, mesh.positions);
});

test("overflow and corrupt poses cannot publish partial geometry or new version stamps", () => {
  const pose = animated(),
    mesh = createAnimationDeformer(pose, geometry());
  const snapshot = () => ({
    positions: [...mesh.positions],
    normals: [...mesh.normals],
    tangents: [...mesh.tangents],
    min: [...mesh.bounds.min],
    max: [...mesh.bounds.max],
    world: [...mesh.worldMatrix],
    version: mesh.version,
    poseVersion: mesh.poseVersion,
  });
  const before = snapshot();
  pose.sample(1);
  pose.morphWeights[0] = 1e308;
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_VALUE"));
  assert.deepEqual(snapshot(), before);
  pose.sample(1);
  pose.jointMatrices[0] = NaN;
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_VALUE"));
  assert.deepEqual(snapshot(), before);
  pose.sample(1);
  pose.worldMatrices[0] = Infinity;
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_VALUE"));
  assert.deepEqual(snapshot(), before);
  pose.sample(1);
  mesh.update();
  near(mesh.positions, [-2, 2, 0]);
});

test("detached pose storage is diagnosed without publishing", () => {
  const pose = animated(),
    mesh = createAnimationDeformer(pose, geometry()),
    before = [...mesh.positions];
  structuredClone(pose.morphWeights.buffer, { transfer: [pose.morphWeights.buffer] });
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_STORAGE"));
  assert.deepEqual([...mesh.positions], before);
  assert.equal(mesh.version, 0);
});

test("detached output bounds fail preflight before other attributes change", () => {
  const pose = animated(),
    mesh = createAnimationDeformer(pose, geometry()),
    before = [...mesh.positions];
  pose.sample(1);
  structuredClone(mesh.bounds.max.buffer, { transfer: [mesh.bounds.max.buffer] });
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_STORAGE"));
  assert.deepEqual([...mesh.positions], before);
  assert.equal(mesh.version, 0);
});

test("deformer disposal is idempotent and never owns its pose", () => {
  const pose = animated(),
    mesh = createAnimationDeformer(pose, geometry());
  mesh.dispose();
  mesh.dispose();
  assert.equal(mesh.disposed, true);
  assert.equal(pose.disposed, false);
  assert.throws(() => mesh.update(), code("ANIMATION_DEFORM_DISPOSED"));
  pose.sample(1);
  const other = createAnimationDeformer(pose, geometry());
  pose.dispose();
  assert.throws(() => other.update(), code("ANIMATION_DISPOSED"));
});

test("invalid skin indices/weights, target shapes, and budgets fail at construction", () => {
  const pose = animated();
  for (const patch of [
    { joints: [1, 0, 0, 0] },
    { weights: [-1, 2, 0, 0] },
    { weights: [0.5, 0, 0, 0] },
    { weights: [0, 0, 0, 0] },
    { influences: 33 },
  ]) {
    assert.throws(
      () => createAnimationDeformer(pose, { ...geometry(), ...patch }),
      code("ANIMATION_DEFORM_SKIN"),
    );
  }
  assert.throws(
    () => createAnimationDeformer(pose, { ...geometry(), morphTargets: [] }),
    code("ANIMATION_DEFORM_MORPH"),
  );
  assert.throws(
    () => createAnimationDeformer(pose, { ...geometry(), morphTargets: [{ colors: [1, 2, 3] }] }),
    code("ANIMATION_DEFORM_MORPH"),
  );
  assert.throws(
    () => createAnimationDeformer(pose, { ...geometry(), tangents: [1, 0, 0, 0] }),
    code("ANIMATION_DEFORM_GEOMETRY"),
  );
  assert.throws(
    () => createAnimationDeformer(pose, { ...geometry(), positions: [NaN, 0, 0] }),
    code("ANIMATION_DEFORM_VALUE"),
  );
  assert.throws(
    () => createAnimationDeformer(pose, geometry(), { maxComponents: 5 }),
    code("ANIMATION_DEFORM_LIMIT"),
  );
  assert.throws(
    () => createAnimationDeformer(pose, { ...geometry(), node: 99 }),
    code("ANIMATION_DEFORM_NODE"),
  );
});

test("plain geometry stays mesh-local and cannot silently accept skin attributes", () => {
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [{ translation: [50, 0, 0] }],
  });
  const mesh = createAnimationDeformer(pose, { node: 0, positions: [1, 2, 3] });
  near(mesh.positions, [1, 2, 3]);
  assert.equal(mesh.worldMatrix[12], 50);
  assert.throws(
    () => createAnimationDeformer(pose, { node: 0, positions: [1, 2, 3], joints: [0] }),
    code("ANIMATION_DEFORM_SKIN"),
  );
});
