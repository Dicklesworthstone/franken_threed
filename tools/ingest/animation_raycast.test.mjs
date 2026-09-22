import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationRaycaster, rayFromAnimationCamera } from "./animation_raycast.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const near = (a, b, e = 1e-8) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) <= e, `${i}: ${a[i]} != ${b[i]}`);
};
const tri = () => [-1, -1, 0, 1, -1, 0, 0, 1, 0];
const ray = (origin = [0, 0, 2], direction = [0, 0, -1], extra = {}) => ({
  origin,
  direction,
  ...extra,
});
function setup(geometries = [{}], options = {}) {
  const pose = { nodeCount: geometries.length, version: 0, disposed: false };
  const descriptors = geometries.map((g, node) => ({
    deformer: {
      node,
      vertexCount: (g.positions ?? tri()).length / 3,
      positions: new Float32Array(g.positions ?? tri()),
      worldMatrix: new Float64Array(g.world ?? identity()),
      version: 0,
      poseVersion: 0,
      disposed: false,
    },
    ...g.descriptor,
  }));
  const caster = createAnimationRaycaster(pose, descriptors, options);
  return { pose, descriptors, caster };
}
function camera(type = "perspective") {
  const projection =
    type === "perspective"
      ? [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0]
      : [0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, -0.1, 0, 0, 0, 0, 1];
  return { type, viewMatrix: identity(), projectionMatrix: projection, cameraPosition: [0, 0, 0] };
}
test("front triangle returns world distance, point, barycentrics, UV and original source IDs", () => {
  const source = { node: 0, mesh: 17, primitive: 8, material: 5 };
  const { caster } = setup([{ descriptor: { texCoords: [0, 0, 1, 0, 0.5, 1], source } }]);
  const hits = caster.raycast(ray([0, 0, 2], [0, 0, -20]));
  assert.equal(hits.length, 1);
  const h = hits[0];
  assert.equal(h.distance, 2);
  near(h.point, [0, 0, 0]);
  near(h.normal, [0, 0, 1]);
  near(h.barycentric, [0.25, 0.25, 0.5]);
  near(h.uv, [0.5, 0.5]);
  assert.equal(h.drawIndex, 0);
  assert.equal(h.faceIndex, 0);
  assert.equal(h.frontFacing, true);
  assert.deepEqual(h.source, source);
  assert.notEqual(h.source, source);
  for (const o of [hits, h, h.point, h.normal, h.barycentric, h.uv, h.source])
    assert.ok(Object.isFrozen(o));
});
test("back-face culling follows material side; double-sided hits report back-facing", () => {
  const single = setup().caster,
    double = setup([{ descriptor: { doubleSided: true } }]).caster;
  assert.equal(single.raycast(ray([0, 0, -2], [0, 0, 1])).length, 0);
  const hit = double.raycast(ray([0, 0, -2], [0, 0, 1]))[0];
  assert.equal(hit.frontFacing, false);
  near(hit.normal, [0, 0, -1]);
});
test("reflected world transforms retain the source front side just like renderer front-face reversal", () => {
  const m = identity();
  m[0] = -2;
  m[5] = 3;
  m[12] = 10;
  const { caster } = setup([{ world: m }]);
  const hit = caster.raycast(ray([10, 0, 4]))[0];
  assert.ok(hit);
  assert.equal(hit.frontFacing, true);
  near(hit.normal, [0, 0, 1]);
  near(hit.barycentric, [0.25, 0.25, 0.5]);
  assert.equal(caster.raycast(ray([10, 0, -4], [0, 0, 1])).length, 0);
});
test("world-space hierarchy handles nonuniform scale, shear, rotation and translation", () => {
  const m = [0, 0, -2, 0, 1, 3, 0, 0, 4, 0, 0, 0, 10, 20, 30, 1];
  const { caster } = setup([{ world: m }]),
    hits = caster.raycast(ray([15, 20, 30], [-3, 0, 0]));
  assert.equal(hits.length, 1);
  near([hits[0].distance], [5]);
  near(hits[0].point, [10, 20, 30]);
  near(hits[0].normal, [3 / Math.sqrt(10), -1 / Math.sqrt(10), 0]);
});
test("singular transforms can leave a queryable planar triangle; collapsed triangles do not hit", () => {
  const m = identity();
  m[10] = 0;
  assert.equal(setup([{ world: m }]).caster.raycast(ray()).length, 1);
  m[0] = 0;
  assert.equal(setup([{ world: m }]).caster.raycast(ray()).length, 0);
});
test("nonindexed and indexed topology preserve source face indices", () => {
  const p = [...tri(), 10, 10, 0, 11, 10, 0, 10, 11, 0];
  const { caster } = setup([
    { positions: p, descriptor: { indices: new Uint16Array([3, 4, 5, 0, 1, 2]) } },
  ]);
  assert.equal(caster.raycast(ray())[0].faceIndex, 1);
  assert.equal(setup([{ positions: p }]).caster.raycast(ray())[0].faceIndex, 0);
});
test("hit ordering is distance then draw then source face, independent of hierarchy traversal", () => {
  const far = identity();
  far[14] = -10;
  const p = [...tri(), ...tri()];
  const { caster } = setup([{ world: far }, { positions: p }, { positions: p }]);
  assert.deepEqual(
    caster.raycast(ray()).map((h) => [h.distance, h.drawIndex, h.faceIndex]),
    [
      [2, 1, 0],
      [2, 1, 1],
      [2, 2, 0],
      [2, 2, 1],
      [12, 0, 0],
    ],
  );
  assert.deepEqual(
    caster
      .raycast(ray(), { firstHitOnly: true })
      .map((h) => [h.distance, h.drawIndex, h.faceIndex]),
    [[2, 1, 0]],
  );
});
test("near/far use normalized world-ray distance with inclusive endpoints", () => {
  const { caster } = setup();
  assert.equal(caster.raycast(ray([0, 0, 2], [0, 0, -100], { near: 2, far: 2 })).length, 1);
  assert.equal(caster.raycast(ray(undefined, undefined, { near: 2.01 })).length, 0);
  assert.equal(caster.raycast(ray(undefined, undefined, { far: 1.99 })).length, 0);
  assert.equal(caster.raycast(ray([0, 0, 0])).length, 1);
});
test("mesh filtering is explicit, unique and does not alter source draw indices", () => {
  const { caster } = setup([{}, {}]);
  assert.deepEqual(
    caster.raycast(ray(), { drawIndices: [1] }).map((h) => h.drawIndex),
    [1],
  );
  assert.deepEqual(caster.raycast(ray(), { drawIndices: [] }), []);
  for (const values of [[0, 0], [-1], [2], null])
    assert.throws(() => caster.raycast(ray(), { drawIndices: values }), {
      code: "ANIMATION_PICK_OPTION",
    });
});
test("all-hit overflow fails instead of returning an arbitrary truncated set; nearest mode remains usable", () => {
  const { caster } = setup([{}, {}]);
  assert.throws(() => caster.raycast(ray(), { maxHits: 1 }), { code: "ANIMATION_PICK_LIMIT" });
  assert.equal(caster.lastQuery, null);
  assert.equal(caster.raycast(ray(), { maxHits: 1, firstHitOnly: true }).length, 1);
});
test("BVH refits on deformer versions, not on every unchanged query, and prior hit snapshots survive", () => {
  const { caster, descriptors, pose } = setup(),
    d = descriptors[0].deformer;
  const old = caster.raycast(ray())[0];
  assert.equal(caster.lastQuery.refittedMeshes, 1);
  caster.raycast(ray());
  assert.equal(caster.lastQuery.refittedMeshes, 0);
  d.worldMatrix[14] = 1;
  d.version++;
  d.poseVersion = ++pose.version;
  const next = caster.raycast(ray())[0];
  assert.equal(next.distance, 1);
  assert.equal(old.distance, 2);
  near(old.point, [0, 0, 0]);
  assert.equal(caster.lastQuery.refittedMeshes, 1);
});
test("deformed position changes refit the tree without rebuilding source face order", () => {
  const { caster, descriptors, pose } = setup(),
    d = descriptors[0].deformer;
  caster.raycast(ray());
  for (let i = 0; i < 3; i++) d.positions[i * 3] += 5;
  d.version++;
  d.poseVersion = ++pose.version;
  assert.equal(caster.raycast(ray()).length, 0);
  assert.equal(caster.raycast(ray([5, 0, 2]))[0].faceIndex, 0);
});
test("stale/partial pose groups fail before publishing any hits or changing query statistics", () => {
  const { caster, descriptors, pose } = setup([{}, {}]);
  caster.raycast(ray());
  const previous = caster.lastQuery;
  pose.version++;
  descriptors[0].deformer.poseVersion = pose.version;
  assert.throws(() => caster.raycast(ray()), { code: "ANIMATION_PICK_STALE" });
  assert.equal(caster.lastQuery, previous);
  descriptors[1].deformer.poseVersion = pose.version;
  assert.equal(caster.raycast(ray()).length, 2);
});
test("failed refit cannot poison later queries or prior snapshots", () => {
  const { caster, descriptors } = setup(),
    d = descriptors[0].deformer;
  const old = caster.raycast(ray())[0],
    last = caster.lastQuery;
  d.positions[0] = NaN;
  d.version++;
  assert.throws(() => caster.raycast(ray()), { code: "ANIMATION_PICK_VALUE" });
  assert.equal(caster.lastQuery, last);
  d.positions[0] = -1;
  assert.equal(caster.raycast(ray())[0].distance, 2);
  near(old.point, [0, 0, 0]);
});
test("detached, shared and resized borrowed storage is rejected", () => {
  const f = setup();
  structuredClone(f.descriptors[0].deformer.positions.buffer, {
    transfer: [f.descriptors[0].deformer.positions.buffer],
  });
  assert.throws(() => f.caster.raycast(ray()), { code: "ANIMATION_PICK_SHAPE" });
  const s = setup();
  s.descriptors[0].deformer.positions = new Float32Array(new SharedArrayBuffer(36));
  assert.throws(() => s.caster.raycast(ray()), { code: "ANIMATION_PICK_STORAGE" });
  const r = setup();
  r.descriptors[0].deformer.positions = new Float32Array(
    new ArrayBuffer(36, { maxByteLength: 72 }),
  );
  assert.throws(() => r.caster.raycast(ray()), { code: "ANIMATION_PICK_STORAGE" });
});
test("ray input capture rejects pose mutation and reentrant raycasts/disposal", () => {
  const f = setup();
  assert.throws(
    () =>
      f.caster.raycast({
        get origin() {
          f.pose.version++;
          return [0, 0, 2];
        },
        direction: [0, 0, -1],
      }),
    { code: "ANIMATION_PICK_CHANGED" },
  );
  const r = setup();
  for (const fn of [() => r.caster.raycast(ray()), () => r.caster.dispose()])
    assert.throws(
      () =>
        r.caster.raycast({
          get origin() {
            fn();
            return [0, 0, 2];
          },
          direction: [0, 0, -1],
        }),
      { code: "ANIMATION_PICK_REENTRANT" },
    );
  assert.equal(r.caster.raycast(ray()).length, 1);
});
test("disposed raycasters/deformers/poses reject queries without disposing borrowed owners", () => {
  const f = setup();
  f.caster.raycast(ray());
  assert.ok(f.caster.bufferBytes > 0);
  f.caster.dispose();
  f.caster.dispose();
  assert.equal(f.caster.bufferBytes, 0);
  assert.equal(f.pose.disposed, false);
  assert.equal(f.descriptors[0].deformer.disposed, false);
  assert.throws(() => f.caster.raycast(ray()), { code: "ANIMATION_PICK_DISPOSED" });
  const p = setup();
  p.pose.disposed = true;
  assert.throws(() => p.caster.raycast(ray()), { code: "ANIMATION_PICK_POSE" });
  const d = setup();
  d.descriptors[0].deformer.disposed = true;
  assert.throws(() => d.caster.raycast(ray()), { code: "ANIMATION_PICK_MESH" });
});
test("topology, UVs, material side and source IDs are snapshotted at construction", () => {
  const indices = [0, 1, 2],
    uv = [0, 0, 1, 0, 0.5, 1],
    source = { node: 0, mesh: 1, primitive: 2, material: 3 };
  const descriptor = { indices, texCoords: uv, source, doubleSided: true },
    f = setup([{ descriptor }]);
  indices.fill(0);
  uv.fill(99);
  source.mesh = 42;
  f.descriptors[0].doubleSided = false;
  const hit = f.caster.raycast(ray([0, 0, -2], [0, 0, 1]))[0];
  assert.ok(hit);
  near(hit.uv, [0.5, 0.5]);
  assert.equal(hit.source.mesh, 1);
});
test("empty topology and an empty mesh group produce an empty immutable hit list", () => {
  const f = setup([{ descriptor: { indices: [] } }]);
  assert.deepEqual(f.caster.raycast(ray()), []);
  assert.deepEqual(setup([]).caster.raycast(ray()), []);
});
for (const [label, value] of [
  ["zero direction", ray(undefined, [0, 0, 0])],
  ["NaN direction", ray(undefined, [NaN, 0, 1])],
  ["nonfinite origin", ray([Infinity, 0, 0])],
  ["negative near", ray(undefined, undefined, { near: -1 })],
  ["invalid far", ray(undefined, undefined, { far: NaN })],
  ["reversed interval", ray(undefined, undefined, { near: 3, far: 2 })],
  ["unknown option", { ...ray(), threshold: 1 }],
])
  test(`rejects ${label}`, () => assert.throws(() => setup().caster.raycast(value)));
for (const [label, geometry, options] of [
  ["invalid index", { descriptor: { indices: [0, 1, 99] } }, {}],
  ["fractional index", { descriptor: { indices: [0, 1, 1.5] } }, {}],
  ["partial triangle", { descriptor: { indices: [0, 1] } }, {}],
  ["invalid UV", { descriptor: { texCoords: [0] } }, {}],
  ["invalid side", { descriptor: { doubleSided: 1 } }, {}],
  [
    "source mismatch",
    { descriptor: { source: { node: 1, mesh: 0, primitive: 0, material: 0 } } },
    {},
  ],
  ["storage budget", {}, { maxBytes: 1 }],
  ["triangle budget", { positions: [...tri(), ...tri()] }, { maxTriangles: 1 }],
])
  test(`construction rejects ${label}`, () => assert.throws(() => setup([geometry], options)));
test("tiny triangles and shared-edge/vertex hits are not lost to a fixed epsilon", () => {
  const tiny = tri().map((x) => x * 1e-20),
    f = setup([{ positions: tiny }]);
  assert.equal(f.caster.raycast(ray([0, 0, 1e-20])).length, 1);
  const square = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
    s = setup([{ positions: square, descriptor: { indices: [0, 1, 2, 0, 2, 3] } }]);
  assert.equal(s.caster.raycast(ray()).length, 2);
  assert.equal(s.caster.raycast(ray([1, 1, 2])).length, 2);
});
test("BVH excludes distant triangles and nearest traversal prunes farther surfaces", () => {
  const positions = [];
  for (let i = 0; i < 4096; i++) positions.push(i * 4 - 1, -1, 0, i * 4 + 1, -1, 0, i * 4, 1, 0);
  const { caster } = setup([{ positions }]);
  assert.equal(caster.raycast(ray(), { firstHitOnly: true })[0].faceIndex, 0);
  assert.ok(caster.lastQuery.trianglesTested <= 8, JSON.stringify(caster.lastQuery));
  assert.ok(caster.lastQuery.boxesTested < 50);
  assert.equal(caster.lastQuery.refittedMeshes, 1);
  assert.equal(caster.raycast(ray([-100, 0, 2])).length, 0);
  assert.equal(caster.lastQuery.trianglesTested, 0);
  assert.equal(caster.lastQuery.refittedMeshes, 0);
});
test("perspective picking supports center/corner rays, lens offsets and camera rotation", () => {
  const c = camera();
  near(rayFromAnimationCamera(c, [0, 0]).direction, [0, 0, -1]);
  near(rayFromAnimationCamera(c, [1, 1]).direction, [2 / 3, 1 / 3, -2 / 3]);
  c.projectionMatrix[8] = 0.2;
  near(rayFromAnimationCamera(c, [0, 0]).direction, [
    0.2 / Math.sqrt(1.04),
    0,
    -1 / Math.sqrt(1.04),
  ]);
  c.projectionMatrix[8] = 0;
  c.cameraPosition = [10, 20, 30];
  c.viewMatrix = [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 30, -20, -10, 1];
  const r = rayFromAnimationCamera(c, [0, 0]);
  near(r.origin, [10, 20, 30]);
  near(r.direction, [-1, 0, 0]);
});
test("orthographic picking offsets the origin but keeps parallel ray directions", () => {
  const c = camera("orthographic");
  c.cameraPosition = [10, 20, 30];
  const a = rayFromAnimationCamera(c, [-1, -1]),
    b = rayFromAnimationCamera(c, [1, 1]);
  near(a.origin, [8, 19, 30]);
  near(b.origin, [12, 21, 30]);
  near(a.direction, [0, 0, -1]);
  near(a.direction, b.direction);
});
test("camera ray preflight rejects unsupported projections, nonrigid views and invalid NDC", () => {
  assert.throws(() => rayFromAnimationCamera(camera(), [2, 0]));
  assert.throws(() => rayFromAnimationCamera(camera(), [NaN, 0]));
  const c = camera();
  c.viewMatrix[0] = 2;
  assert.throws(() => rayFromAnimationCamera(c, [0, 0]), { code: "ANIMATION_PICK_CAMERA" });
  const p = camera();
  p.projectionMatrix[1] = 1;
  assert.throws(() => rayFromAnimationCamera(p, [0, 0]), { code: "ANIMATION_PICK_CAMERA" });
});
// Independent Moller-Trumbore reference (not the production shear-edge test).
function oracle(p, o, d) {
  const sub = (a, b) => a.map((x, i) => x - b[i]),
    cross = (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ],
    dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const a = p.slice(0, 3),
    b = p.slice(3, 6),
    c = p.slice(6, 9),
    e1 = sub(b, a),
    e2 = sub(c, a),
    h = cross(d, e2),
    det = dot(e1, h);
  if (Math.abs(det) < 1e-14) return null;
  const s = sub(o, a),
    u = dot(s, h) / det;
  if (u < 0 || u > 1) return null;
  const q = cross(s, e1),
    v = dot(d, q) / det;
  if (v < 0 || u + v > 1) return null;
  const t = dot(e2, q) / det;
  return t >= 0 ? t : null;
}
test("deterministic oblique rays match an independent brute-force triangle oracle", () => {
  let seed = 90210;
  const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const p = [];
  for (let i = 0; i < 256; i++)
    for (let j = 0; j < 9; j++) p.push(Math.fround((random() - 0.5) * 20));
  const f = setup([{ positions: p, descriptor: { doubleSided: true } }]);
  for (let i = 0; i < 100; i++) {
    const o = [(random() - 0.5) * 30, (random() - 0.5) * 30, 20],
      raw = [(random() - 0.5) * 0.5, (random() - 0.5) * 0.5, -1],
      n = Math.hypot(...raw),
      d = raw.map((x) => x / n),
      expected = [];
    for (let face = 0; face < 256; face++) {
      const distance = oracle(p.slice(face * 9, face * 9 + 9), o, d);
      if (distance !== null) expected.push({ distance, face });
    }
    expected.sort((a, b) => a.distance - b.distance || a.face - b.face);
    const actual = f.caster.raycast(ray(o, d));
    assert.deepEqual(
      actual.map((h) => h.faceIndex),
      expected.map((h) => h.face),
    );
    near(
      actual.map((h) => h.distance),
      expected.map((h) => h.distance),
      1e-8,
    );
    const first = f.caster.raycast(ray(o, d), { firstHitOnly: true });
    assert.deepEqual(first, actual.slice(0, 1));
  }
});
