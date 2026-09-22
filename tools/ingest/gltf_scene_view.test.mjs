import assert from "node:assert/strict";
import test from "node:test";
import { createGltfSceneView, decodeGltfSceneView } from "./gltf_scene_view.mjs";

const near = (a, b, epsilon = 1e-10) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) <= epsilon, `${i}: ${a[i]} != ${b[i]}`);
};
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const transform = (m, p) =>
  [0, 1, 2, 3].map(
    (r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * (p[3] ?? 1),
  );
const ndc = (m, p) => {
  const v = transform(m, p);
  return v.slice(0, 3).map((x) => x / v[3]);
};
function fixture({ type = "perspective", projection, lights = [] } = {}) {
  const camera = {
    type,
    [type]:
      projection ??
      (type === "perspective"
        ? { yfov: Math.PI / 2, znear: 1, zfar: 11 }
        : { xmag: 2, ymag: 1, znear: 0, zfar: 10 }),
  };
  const model = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [
      { children: [1, ...lights.map((_, i) => i + 2)] },
      { camera: 0, name: "view node" },
      ...lights.map((_, light) => ({ extensions: { KHR_lights_punctual: { light } } })),
    ],
    cameras: [camera],
    extensions: { KHR_lights_punctual: { lights } },
  };
  const pose = {
    nodeCount: model.nodes.length,
    worldMatrices: new Float64Array(model.nodes.flatMap(identity)),
    version: 0,
    disposed: false,
  };
  const definition = decodeGltfSceneView(model),
    view = createGltfSceneView(pose, definition);
  return { model, pose, definition, view };
}
test("finite perspective maps the near/far planes and viewport edges into WebGPU clip space", () => {
  const { view } = fixture(),
    s = view.sample({ aspectRatio: 2 }),
    p = s.projectionMatrix;
  near(ndc(p, [0, 0, -1]), [0, 0, 0]);
  near(ndc(p, [0, 0, -11]), [0, 0, 1]);
  near(ndc(p, [2, 1, -1]), [1, 1, 0]);
  assert.equal(s.cameraNode, 1);
  assert.equal(s.cameraIndex, 0);
  assert.equal(s.type, "perspective");
  assert.equal(s.poseVersion, 0);
  near(s.viewProjection, p);
});
test("infinite perspective does not invent a finite far clipping plane", () => {
  const { view } = fixture({ projection: { yfov: Math.PI / 2, znear: 0.25 } }),
    s = view.sample({ aspectRatio: 1 });
  near(ndc(s.viewProjection, [0, 0, -0.25]), [0, 0, 0]);
  near(ndc(s.viewProjection, [0, 0, -1e12]), [0, 0, 1], 1e-12);
  assert.equal(s.projectionMatrix[10], -1);
  assert.equal(s.projectionMatrix[14], -0.25);
});
test("orthographic projection preserves extents, zero near and constant view direction", () => {
  const { view } = fixture({ type: "orthographic" }),
    s = view.sample();
  near(ndc(s.viewProjection, [2, 1, 0]), [1, 1, 0]);
  near(ndc(s.viewProjection, [-2, -1, -10]), [-1, -1, 1]);
  assert.equal(s.lighting.cameraPosition, undefined);
  near(s.lighting.viewDirection, [0, 0, 1]);
});
test("implicit aspect must be supplied; authored aspect remains authoritative", () => {
  const f = fixture();
  assert.throws(() => f.view.sample(), { code: "GLTF_VIEW_ASPECT" });
  near(f.view.sample({ aspectRatio: 1 }).projectionMatrix.slice(0, 1), [1]);
  near(f.view.sample({ aspectRatio: 2 }).projectionMatrix.slice(0, 1), [0.5]);
  const { view } = fixture({ projection: { yfov: Math.PI / 2, znear: 1, aspectRatio: 3 } });
  near(view.sample().projectionMatrix.slice(0, 1), [1 / 3]);
  near(view.sample({ aspectRatio: 1 }).projectionMatrix.slice(0, 1), [1 / 3]);
});
test("world camera pose and nonuniform scale do not scale the rendered view", () => {
  const { view, pose } = fixture();
  // Rotation +90 degrees about Y, nonuniform scale, translated world position.
  const m = [0, 0, -2, 0, 0, 3, 0, 0, 4, 0, 0, 0, 10, 20, 30, 1];
  pose.worldMatrices.set(m, 16);
  const s = view.sample({ aspectRatio: 1 });
  near(s.cameraPosition, [10, 20, 30]);
  near(s.viewDirection, [1, 0, 0]);
  near(transform(s.viewMatrix, [9, 20, 30]), [0, 0, -1, 1]);
  near(ndc(s.viewProjection, [9, 20, 30]), [0, 0, 0]);
  near(transform(s.viewMatrix, [10, 21, 30]), [0, 1, 0, 1]);
});
test("inherited shear is orthonormalized without changing the world lens axis", () => {
  const { view, pose } = fixture(),
    m = identity();
  m[4] = 1;
  m[5] = 2;
  m[6] = 1;
  m[8] = 1;
  m[10] = 2;
  pose.worldMatrices.set(m, 16);
  const s = view.sample({ aspectRatio: 1 }),
    v = s.viewMatrix;
  const rows = [0, 1, 2].map((r) => [v[r], v[r + 4], v[r + 8]]);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      near([rows[i].reduce((sum, x, k) => sum + x * rows[j][k], 0)], [i === j ? 1 : 0]);
  near(s.viewDirection, [1 / Math.sqrt(5), 0, 2 / Math.sqrt(5)]);
});
test("directional, point and spot properties map to existing renderer world-space lighting", () => {
  const input = [
    { type: "directional", color: [0.1, 0.2, 0.3], intensity: 2 },
    { type: "point", range: 7, intensity: 3 },
    { type: "spot", range: 9, spot: { innerConeAngle: 0.2, outerConeAngle: 0.7 } },
  ];
  const { view, pose } = fixture({ lights: input });
  for (let n = 2; n < 5; n++) {
    const m = [0, 0, -2, 0, 0, 4, 0, 0, 8, 0, 0, 0, n, 10, 20, 1];
    pose.worldMatrices.set(m, n * 16);
  }
  const lights = view.sample({ aspectRatio: 1 }).lighting.lights;
  near(lights[0].direction, [-1, 0, 0]);
  assert.equal(lights[0].position, undefined);
  assert.equal(lights[0].intensity, 2);
  near(lights[0].color, [0.1, 0.2, 0.3]);
  near(lights[1].position, [3, 10, 20]);
  assert.equal(lights[1].direction, undefined);
  assert.equal(lights[1].range, 7);
  near(lights[2].position, [4, 10, 20]);
  near(lights[2].direction, [-1, 0, 0]);
  assert.equal(lights[2].range, 9);
  assert.equal(lights[2].innerConeAngle, 0.2);
  assert.equal(lights[2].outerConeAngle, 0.7);
  assert.deepEqual(view.sampleLights(), lights);
});
test("point lights with collapsed scale remain valid; directional zero forward axes are refused", () => {
  const f = fixture({ lights: [{ type: "point" }] }),
    m = identity();
  m[0] = m[5] = m[10] = 0;
  m[12] = 3;
  f.pose.worldMatrices.set(m, 32);
  near(f.view.sampleLights()[0].position, [3, 0, 0]);
  const d = fixture({ lights: [{ type: "directional" }] });
  d.pose.worldMatrices.set(m, 32);
  assert.throws(() => d.view.sampleLights(), { code: "GLTF_VIEW_TRANSFORM" });
});
test("snapshots retain previous frame matrices/lights while current pose advances", () => {
  const { view, pose } = fixture({ lights: [{ type: "point" }] }),
    first = view.sample({ aspectRatio: 1 });
  pose.worldMatrices[28] = 8;
  pose.worldMatrices[44] = 9;
  pose.version++;
  const next = view.sample({ aspectRatio: 1 });
  assert.equal(next.poseVersion, 1);
  assert.equal(first.poseVersion, 0);
  near(first.cameraPosition, [0, 0, 0]);
  near(first.lighting.lights[0].position, [0, 0, 0]);
  near(next.cameraPosition, [8, 0, 0]);
  near(next.lighting.lights[0].position, [9, 0, 0]);
  for (const obj of [
    first,
    first.viewProjection,
    first.lighting,
    first.lighting.lights,
    first.lighting.lights[0].position,
  ])
    assert.ok(Object.isFrozen(obj));
});
test("shared camera/light definitions retain selected scene instance and source IDs", () => {
  const f = fixture({ lights: [{ type: "point" }] }),
    m = f.model;
  m.nodes.push({ camera: 0, extensions: { KHR_lights_punctual: { light: 0 } } }, { camera: 999 });
  m.nodes[0].children.push(3);
  m.scenes.push({ nodes: [4] });
  const d = decodeGltfSceneView(m);
  assert.deepEqual(
    d.cameras.map((c) => [c.node, c.camera]),
    [
      [1, 0],
      [3, 0],
    ],
  );
  assert.deepEqual(
    d.lights.map((l) => [l.node, l.light]),
    [
      [2, 0],
      [3, 0],
    ],
  );
  const pose = {
    nodeCount: m.nodes.length,
    version: 0,
    worldMatrices: new Float64Array(m.nodes.flatMap(identity)),
  };
  const v = createGltfSceneView(pose, d);
  assert.throws(() => v.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_CAMERA" });
  assert.equal(v.sample({ cameraNode: 3, aspectRatio: 1 }).cameraNode, 3);
  assert.throws(() => v.sample({ cameraNode: 0, aspectRatio: 1 }), { code: "GLTF_VIEW_CAMERA" });
  assert.throws(() => decodeGltfSceneView(m, { scene: 1 }), { code: "GLTF_VIEW_INDEX" });
});
test("source metadata is snapshotted and unused cameras/lights are never instantiated", () => {
  const { view, model } = fixture({ lights: [{ type: "directional" }] });
  model.cameras[0].perspective.yfov = 0;
  model.extensions.KHR_lights_punctual.lights[0].intensity = 99;
  assert.equal(view.sampleLights()[0].intensity, 1);
  near(view.sample({ aspectRatio: 1 }).projectionMatrix.slice(0, 1), [1]);
  const plain = {
    asset: { version: "2.0" },
    scenes: [{ nodes: [] }],
    cameras: [{ type: "bogus" }],
    extensions: { KHR_lights_punctual: { lights: [{ type: "bogus" }] } },
  };
  const d = decodeGltfSceneView(plain),
    v = createGltfSceneView({ nodeCount: 0, version: 0 }, d);
  assert.deepEqual(v.sampleLights(), []);
  assert.throws(() => v.sample(), { code: "GLTF_VIEW_CAMERA" });
});
for (const [name, mutate] of [
  [
    "zero near",
    (p) => {
      p.znear = 0;
    },
  ],
  [
    "bad far",
    (p) => {
      p.zfar = 0.5;
    },
  ],
  [
    "zero yfov",
    (p) => {
      p.yfov = 0;
    },
  ],
  [
    "PI yfov",
    (p) => {
      p.yfov = Math.PI;
    },
  ],
  [
    "bad aspect",
    (p) => {
      p.aspectRatio = -1;
    },
  ],
  [
    "nonfinite",
    (p) => {
      p.yfov = NaN;
    },
  ],
  [
    "extended projection",
    (p) => {
      p.extensions = { EXT_unknown: {} };
    },
  ],
])
  test(`invalid camera ${name} fails preflight`, () => {
    const f = fixture();
    mutate(f.model.cameras[0].perspective);
    assert.throws(() => decodeGltfSceneView(f.model));
  });
for (const light of [
  { type: "ambient" },
  { type: "point", intensity: -1 },
  { type: "point", color: [2, 1, 1] },
  { type: "point", range: 0 },
  { type: "directional", range: 1 },
  { type: "spot" },
  { type: "spot", spot: { innerConeAngle: 0.5, outerConeAngle: 0.5 } },
  { type: "spot", spot: { outerConeAngle: 2 } },
  { type: "point", spot: {} },
  { type: "point", extensions: { EXT_shadow: {} } },
])
  test(`invalid punctual light fails preflight: ${JSON.stringify(light)}`, () => {
    assert.throws(() => fixture({ lights: [light] }));
  });
test("all eight source lights are retained; excess lights fail rather than disappear", () => {
  assert.equal(
    fixture({ lights: Array.from({ length: 8 }, () => ({ type: "point" })) }).view.sampleLights()
      .length,
    8,
  );
  assert.throws(() => fixture({ lights: Array.from({ length: 9 }, () => ({ type: "point" })) }), {
    code: "GLTF_VIEW_LIMIT",
  });
});
for (const [label, nodes, roots] of [
  ["cycle", [{ children: [1] }, { children: [0] }], []],
  ["duplicate child", [{ children: [1, 1] }, {}], [0]],
  ["multiple parents", [{ children: [2] }, { children: [2] }, {}], [0, 1]],
  ["non-root selection", [{ children: [1] }, {}], [1]],
  ["duplicate root", [{}], [0, 0]],
  ["invalid child", [{ children: [2] }], [0]],
])
  test(`rejects ${label}`, () => {
    assert.throws(() =>
      decodeGltfSceneView({ asset: { version: "2.0" }, nodes, scenes: [{ nodes: roots }] }),
    );
  });
test("hierarchy traversal is iterative at deep valid depth", () => {
  const nodes = Array.from({ length: 12000 }, (_, i) =>
    i < 11999 ? { children: [i + 1] } : { camera: 0 },
  );
  const model = {
    asset: { version: "2.0" },
    nodes,
    scenes: [{ nodes: [0] }],
    cameras: [{ type: "perspective", perspective: { yfov: 1, znear: 1 } }],
  };
  assert.equal(decodeGltfSceneView(model).cameras[0].node, 11999);
});
test("failed samples leave old snapshots untouched and permit recovery", () => {
  const { view, pose } = fixture(),
    good = view.sample({ aspectRatio: 1 });
  pose.worldMatrices[26] = 0;
  assert.throws(() => view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_TRANSFORM" });
  near(good.cameraPosition, [0, 0, 0]);
  pose.worldMatrices[26] = 1;
  assert.deepEqual(view.sample({ aspectRatio: 1 }), good);
  pose.worldMatrices[19] = 0.1;
  assert.throws(() => view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_TRANSFORM" });
  pose.worldMatrices[19] = 0;
  pose.worldMatrices[28] = Infinity;
  assert.throws(() => view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_VALUE" });
});
test("disposed, resized and detached pose storage never produces a frame", () => {
  const f = fixture();
  f.pose.disposed = true;
  assert.throws(() => f.view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_POSE" });
  f.pose.disposed = false;
  f.pose.worldMatrices = new Float64Array(0);
  assert.throws(() => f.view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_POSE" });
  const d = fixture();
  structuredClone(d.pose.worldMatrices.buffer, { transfer: [d.pose.worldMatrices.buffer] });
  assert.throws(() => d.view.sample({ aspectRatio: 1 }), { code: "GLTF_VIEW_POSE" });
});
test("reentrant camera options and pose mutation are rejected without advancing the pose", () => {
  const f = fixture();
  assert.throws(
    () =>
      f.view.sample({
        get aspectRatio() {
          f.view.sample({ aspectRatio: 1 });
          return 1;
        },
      }),
    { code: "GLTF_VIEW_REENTRANT" },
  );
  assert.equal(f.pose.version, 0);
  assert.throws(
    () =>
      f.view.sample({
        get aspectRatio() {
          f.pose.version++;
          return 1;
        },
      }),
    { code: "GLTF_VIEW_CHANGED" },
  );
  assert.equal(f.view.sample({ aspectRatio: 1 }).poseVersion, 1);
});
test("lower-level public metadata is revalidated rather than trusted", () => {
  const f = fixture(),
    bad = structuredClone(f.definition);
  bad.cameras[0].projection.znear = 0;
  assert.throws(() => createGltfSceneView(f.pose, bad));
  const invalid = structuredClone(f.definition);
  invalid.cameras[0].node = 999;
  assert.throws(() => createGltfSceneView(f.pose, invalid));
  const shared = structuredClone(f.definition),
    v = createGltfSceneView(f.pose, shared);
  shared.cameras[0].projection.yfov = 0;
  assert.equal(v.sample({ aspectRatio: 1 }).cameraNode, 1);
});
test("many rotated/scaled world cameras transform independently constructed view points correctly", () => {
  const { view, pose } = fixture();
  for (let i = 1; i <= 64; i++) {
    const a = i * 0.137,
      c = Math.cos(a),
      s = Math.sin(a),
      position = [i, -i / 3, 2 * i];
    pose.worldMatrices.set(
      [c * 2, 0, -s * 2, 0, 0, 5, 0, 0, s * 7, 0, c * 7, 0, ...position, 1],
      16,
    );
    const point = [position[0] + c * 0.3 - s * 4, position[1] + 0.5, position[2] - s * 0.3 - c * 4];
    const frame = view.sample({ aspectRatio: 1 });
    near(transform(frame.viewMatrix, point), [0.3, 0.5, -4, 1], 1e-12);
  }
});
