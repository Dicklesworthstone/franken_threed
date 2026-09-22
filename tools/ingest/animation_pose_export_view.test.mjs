import assert from "node:assert/strict";
import test from "node:test";
import { exportAnimationPoseGLB } from "./animation_pose_export.mjs";
import { createGltfSceneView, decodeGltfSceneView } from "./gltf_scene_view.mjs";

// Actual writer + scene-view evaluation, with independent GLB/layout and TRS
// reconstruction. No native renderer, GPU, image decoder or animation clock.
const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const near = (a, b, epsilon = 2e-12) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(
      Math.abs(a[i] - b[i]) <= epsilon * Math.max(1, Math.abs(a[i]), Math.abs(b[i])),
      `${i}: ${a[i]} != ${b[i]}`,
    );
};
const camera = (node = 0, projection = { yfov: 0.9, znear: 0.1 }, other = {}) => ({
  node,
  camera: 4,
  name: "lens",
  nodeName: "camera node",
  type: "perspective",
  projection,
  ...other,
});
const light = (node = 0, type = "point", other = {}) => ({
  node,
  light: 5,
  name: type + " lamp",
  nodeName: "lamp node",
  type,
  color: [0.3, 0.5, 0.7],
  intensity: 8,
  ...other,
});
function fixture({ cameras = [], lights = [], matrices = [I()] } = {}) {
  const pose = {
    version: 3,
    disposed: false,
    nodeCount: matrices.length,
    worldMatrices: Float64Array.from(matrices.flat()),
  };
  const sceneView = {
    format: "f3d-gltf-scene-view-v1",
    nodeCount: pose.nodeCount,
    cameras,
    lights,
  };
  const entry = {
    source: { node: 0, mesh: 2, primitive: 1, material: 0 },
    deformer: {
      vertexCount: 3,
      poseVersion: 3,
      disposed: false,
      worldMatrix: I(),
      positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
    },
    drawable: { shading: "unlit" },
  };
  return { pose, sceneView, entry };
}
function parse(buffer) {
  const v = new DataView(buffer),
    bytes = new Uint8Array(buffer);
  assert.equal(v.getUint32(0, true), 0x46546c67);
  assert.equal(v.getUint32(4, true), 2);
  assert.equal(v.getUint32(8, true), bytes.length);
  const length = v.getUint32(12, true);
  assert.equal(length % 4, 0);
  assert.equal(v.getUint32(16, true), 0x4e4f534a);
  assert.equal(v.getUint32(24 + length, true), 0x004e4942);
  assert.equal(v.getUint32(20 + length, true), bytes.length - 28 - length);
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + length)));
  for (const node of json.nodes) {
    assert.equal(node.children, undefined);
    if (node.matrix)
      for (const key of ["translation", "rotation", "scale"]) assert.equal(node[key], undefined);
  }
  assert.deepEqual(
    json.scenes[0].nodes,
    json.nodes.map((_, i) => i),
  );
  return { json, buffer };
}
const run = async (f, options = {}) =>
  parse(await exportAnimationPoseGLB(f.pose, [f.entry], { sceneView: f.sceneView, ...options }));
function matrix(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1],
    t = node.translation ?? [0, 0, 0];
  assert.equal(node.scale, undefined);
  if (node.rotation) near([Math.hypot(x, y, z, w)], [1]);
  return [
    1 - 2 * y * y - 2 * z * z,
    2 * x * y + 2 * z * w,
    2 * x * z - 2 * y * w,
    0,
    2 * x * y - 2 * z * w,
    1 - 2 * x * x - 2 * z * z,
    2 * y * z + 2 * x * w,
    0,
    2 * x * z + 2 * y * w,
    2 * y * z - 2 * x * w,
    1 - 2 * x * x - 2 * y * y,
    0,
    ...t,
    1,
  ];
}
function reimport(json) {
  const pose = {
    version: 3,
    disposed: false,
    nodeCount: json.nodes.length,
    worldMatrices: Float64Array.from(json.nodes.flatMap(matrix)),
  };
  return createGltfSceneView(pose, decodeGltfSceneView(json));
}
function compareCameras(f, r, aspect = 1.7) {
  const before = createGltfSceneView(f.pose, f.sceneView),
    after = reimport(r.json);
  assert.equal(before.cameras.length, after.cameras.length);
  for (let i = 0; i < before.cameras.length; i++) {
    const a = before.sample({ cameraNode: before.cameras[i].node, aspectRatio: aspect });
    const b = after.sample({ cameraNode: after.cameras[i].node, aspectRatio: aspect });
    for (const key of [
      "viewMatrix",
      "projectionMatrix",
      "viewProjection",
      "cameraPosition",
      "viewDirection",
    ])
      near(a[key], b[key]);
  }
}
function compareLights(f, r) {
  const before = createGltfSceneView(f.pose, f.sceneView).sampleLights(),
    after = reimport(r.json).sampleLights();
  assert.equal(before.length, after.length);
  for (let i = 0; i < before.length; i++)
    for (const key of Object.keys(before[i])) {
      if (Array.isArray(before[i][key])) near(before[i][key], after[i][key]);
      else assert.equal(before[i][key], after[i][key]);
    }
}
test("perspective camera preserves authored projection, names, origin and static geometry", async () => {
  const m = I();
  m.splice(12, 3, 4, 5, 6);
  const f = fixture({
      cameras: [camera(0, { yfov: 0.7, znear: 0.2, zfar: 100, aspectRatio: 2 })],
      matrices: [m],
    }),
    r = await run(f);
  assert.deepEqual(r.json.cameras, [
    {
      type: "perspective",
      perspective: { yfov: 0.7, znear: 0.2, zfar: 100, aspectRatio: 2 },
      name: "lens",
    },
  ]);
  const n = r.json.nodes[1];
  assert.equal(n.camera, 0);
  assert.equal(n.name, "camera node");
  assert.deepEqual(n.extras.f3dSource, { node: 0, camera: 4 });
  near(n.matrix, m);
  compareCameras(f, r);
  assert.equal(r.json.meshes.length, 1);
  const a = r.json.accessors[r.json.meshes[0].primitives[0].attributes.POSITION];
  assert.deepEqual(a.min, [0, 0, 0]);
  assert.deepEqual(a.max, [1, 1, 0]);
  assert.equal(r.json.animations, undefined);
  assert.equal(r.json.skins, undefined);
});
test("unspecified aspect and infinite far remain unspecified, not frozen to export viewport", async () => {
  const f = fixture({ cameras: [camera()] }),
    r = await run(f);
  assert.deepEqual(r.json.cameras[0].perspective, { yfov: 0.9, znear: 0.1 });
  for (const aspect of [0.5, 1, 2.7]) compareCameras(f, r, aspect);
  assert.throws(() => reimport(r.json).sample(), { code: "GLTF_VIEW_ASPECT" });
});
test("orthographic magnitudes and finite clip planes are preserved", async () => {
  const f = fixture({
      cameras: [camera(0, { xmag: 4, ymag: 3, znear: 0, zfar: 80 }, { type: "orthographic" })],
    }),
    r = await run(f);
  assert.deepEqual(r.json.cameras[0].orthographic, { xmag: 4, ymag: 3, znear: 0, zfar: 80 });
  compareCameras(f, r);
});
for (const [label, m] of [
  ["nonuniform scale", [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1]],
  ["inherited shear", [2, 0, 1, 0, 1, 3, 0.4, 0, 0.3, 0.2, 4, 0, 5, 6, 7, 1]],
  ["reflected basis", [-2, 0, 0, 0, 0, 3, 0, 0, 0, 0, -4, 0, 3, 2, 1, 1]],
])
  test(`exports a rigid camera matching the existing ${label} view profile`, async () => {
    const f = fixture({ cameras: [camera()], matrices: [m] }),
      r = await run(f);
    compareCameras(f, r);
    const a = r.json.nodes[1].matrix,
      x = a.slice(0, 3),
      y = a.slice(4, 7),
      z = a.slice(8, 11);
    near([Math.hypot(...x), Math.hypot(...y), Math.hypot(...z)], [1, 1, 1]);
    near(
      [
        x.reduce((s, n, i) => s + n * y[i], 0),
        x.reduce((s, n, i) => s + n * z[i], 0),
        y.reduce((s, n, i) => s + n * z[i], 0),
      ],
      [0, 0, 0],
    );
    const cross = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
    near(cross, z);
  });
test("point, directional and spot lights preserve photometry and current-pose world placement", async () => {
  const a = I(),
    b = I(),
    c = I();
  a.splice(12, 3, 10, 20, 30);
  b.splice(12, 3, -1, -2, -3);
  c.splice(12, 3, 8, 9, 10);
  a[0] = a[5] = a[10] = 3;
  b.splice(8, 3, 2, -3, -4);
  c.splice(8, 3, -3, 1, 2);
  const f = fixture({
      matrices: [a, b, c],
      lights: [
        light(0, "point", { range: 50 }),
        light(1, "directional"),
        light(2, "spot", { range: 8, innerConeAngle: 0.1, outerConeAngle: 0.6 }),
      ],
    }),
    r = await run(f);
  compareLights(f, r);
  assert.deepEqual(r.json.extensionsUsed, ["KHR_materials_unlit", "KHR_lights_punctual"]);
  assert.deepEqual(r.json.extensionsRequired, r.json.extensionsUsed);
  const defs = r.json.extensions.KHR_lights_punctual.lights;
  assert.equal(defs[0].range, 50);
  assert.equal(defs[1].range, undefined);
  assert.deepEqual(defs[2].spot, { innerConeAngle: 0.1, outerConeAngle: 0.6 });
  for (const d of defs) assert.equal(d.intensity, 8);
  near(r.json.nodes[2].translation, [-1, -2, -3]);
  assert.equal(r.json.nodes[1].rotation, undefined);
});
test("one source node hosting camera and spot light becomes one rigid output instance", async () => {
  const m = [2, 0, 1, 0, 0.7, 3, 0.2, 0, 1, 0.5, 2, 0, 7, 8, 9, 1];
  const f = fixture({
      matrices: [m],
      cameras: [camera()],
      lights: [light(0, "spot", { innerConeAngle: 0, outerConeAngle: 0.5 })],
    }),
    r = await run(f);
  assert.equal(r.json.nodes.length, 2);
  assert.equal(r.json.nodes[1].camera, 0);
  assert.equal(r.json.nodes[1].extensions.KHR_lights_punctual.light, 0);
  assert.deepEqual(r.json.nodes[1].extras.f3dSource, { node: 0, camera: 4, light: 5 });
  compareCameras(f, r);
  compareLights(f, r);
});
test("shared camera/light definitions deduplicate while repeated node instances remain distinct", async () => {
  const m = I();
  m[12] = 17;
  const f = fixture({
      matrices: [I(), m],
      cameras: [camera(0), camera(1)],
      lights: [light(0, "directional"), light(1, "directional")],
    }),
    r = await run(f);
  assert.equal(r.json.cameras.length, 1);
  assert.equal(r.json.extensions.KHR_lights_punctual.lights.length, 1);
  assert.equal(r.json.nodes.length, 3);
  assert.equal(r.json.nodes[1].camera, r.json.nodes[2].camera);
  assert.equal(r.json.nodes[2].matrix[12], 17);
  compareCameras(f, r);
  compareLights(f, r);
});
test("only decoded selected-scene instances are included, not unused definitions or other scenes", async () => {
  const model = {
    asset: { version: "2.0" },
    scene: 1,
    scenes: [{ nodes: [0] }, { nodes: [1] }],
    nodes: [{ camera: 0 }, { camera: 1, extensions: { KHR_lights_punctual: { light: 0 } } }],
    cameras: [
      { type: "perspective", perspective: { yfov: 0.5, znear: 0.1 } },
      { type: "perspective", perspective: { yfov: 1.2, znear: 0.3 } },
    ],
    extensions: {
      KHR_lights_punctual: { lights: [{ type: "point" }, { type: "point", intensity: 99 }] },
    },
  };
  const f = fixture({ matrices: [I(), I()] });
  f.sceneView = decodeGltfSceneView(model);
  const r = await run(f);
  assert.equal(r.json.cameras.length, 1);
  assert.equal(r.json.cameras[0].perspective.yfov, 1.2);
  assert.equal(r.json.extensions.KHR_lights_punctual.lights.length, 1);
  assert.equal(r.json.nodes[1].extras.f3dSource.node, 1);
});
test("live view metadata is accepted and empty or explicitly absent views add no scene resources", async () => {
  const f = fixture({ cameras: [camera()] });
  const view = createGltfSceneView(f.pose, f.sceneView),
    r = await run(f, { sceneView: view });
  compareCameras(f, r);
  for (const sceneView of [null, undefined, fixture().sceneView]) {
    const r = await run(f, { sceneView });
    assert.equal(r.json.nodes.length, 1);
    assert.equal(r.json.cameras, undefined);
    assert.equal(r.json.extensions, undefined);
  }
});
test("invalid scene metadata and degenerate source frames fail before any image resolver", async () => {
  for (const alter of [
    (f) => f.sceneView.nodeCount++,
    (f) => (f.sceneView.cameras[0].node = 99),
    (f) => f.sceneView.cameras.push({ ...f.sceneView.cameras[0] }),
    (f) => (f.sceneView.cameras[0].projection.znear = -1),
    (f) => (f.pose.worldMatrices[10] = 0),
    (f) => (f.pose.worldMatrices[5] = 0),
    (f) => (f.sceneView.lights = [light(0, "spot", { innerConeAngle: 0.5, outerConeAngle: 0.1 })]),
  ]) {
    const f = fixture({ cameras: [camera()] });
    f.entry.drawable.baseColorTexture = { view: {}, sampler: {} };
    f.entry.drawable.texCoords = [0, 0, 1, 0, 0, 1];
    alter(f);
    let calls = 0;
    await assert.rejects(
      run(f, {
        resolveTexture: () => {
          calls++;
        },
      }),
      (e) => e.code?.startsWith("GLTF_VIEW_"),
    );
    assert.equal(calls, 0);
  }
});
test("scene metadata and pose are captured before asynchronous image resolution", async () => {
  const f = fixture({ cameras: [camera()], lights: [light()] });
  f.entry.drawable.baseColorTexture = { view: {}, sampler: {} };
  f.entry.drawable.texCoords = [0, 0, 1, 0, 0, 1];
  let finish;
  const pending = run(f, {
    resolveTexture: () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  });
  f.pose.worldMatrices[12] = 123;
  f.pose.version++;
  f.pose.disposed = true;
  f.entry.deformer.disposed = true;
  f.sceneView.cameras[0].projection.yfov = 2;
  f.sceneView.lights[0].intensity = 100;
  finish({ bytes: Uint8Array.of(255, 216, 255, 217), mimeType: "image/jpeg" });
  const r = await pending;
  assert.equal(r.json.cameras[0].perspective.yfov, 0.9);
  assert.equal(r.json.nodes[1].matrix[12], 0);
  assert.equal(r.json.extensions.KHR_lights_punctual.lights[0].intensity, 8);
  assert.equal(r.json.extras.f3d.poseVersion, 3);
});
test("camera/light JSON and root nodes count toward exact final-file budget", async () => {
  const f = fixture({ cameras: [camera()], lights: [light()] }),
    r = await run(f);
  assert.equal(
    (await run(f, { maxBytes: r.buffer.byteLength })).buffer.byteLength,
    r.buffer.byteLength,
  );
  await assert.rejects(run(f, { maxBytes: r.buffer.byteLength - 1 }), {
    code: "ANIMATION_EXPORT_LIMIT",
  });
});
test("deterministic randomized light-direction round trips include axial and antipodal cases", async () => {
  let state = 0x137991ab;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state / 0x100000000) * 2 - 1;
  };
  const directions = [
    [0, 0, 1],
    [0, 0, -1],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [1e-15, 0, -1],
    [-1e-15, 1e-15, 1],
  ];
  for (let i = 0; i < 1000; i++) directions.push([random(), random(), random()]);
  for (const axis of directions) {
    const m = I();
    m.splice(8, 3, ...axis);
    const f = fixture({ matrices: [m], lights: [light(0, "directional")] }),
      r = await run(f);
    compareLights(f, r);
  }
});
test("current pose invalidation during view capture cannot publish a mixed-version export", async () => {
  const f = fixture({ cameras: [camera()] });
  const original = f.pose.worldMatrices;
  let reads = 0;
  Object.defineProperty(f.pose, "worldMatrices", {
    get() {
      if (++reads === 1) f.pose.version++;
      return original;
    },
  });
  await assert.rejects(run(f), { code: "GLTF_VIEW_CHANGED" });
});
