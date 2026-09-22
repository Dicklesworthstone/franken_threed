import assert from "node:assert/strict";
import test from "node:test";
import { decodeGltfGeometry } from "./animation_geometry.mjs";
import { decodeGltfAnimation } from "./animation_gltf.mjs";

const types = {
  5120: [1, "setInt8", -128, 127],
  5121: [1, "setUint8", 0, 255],
  5122: [2, "setInt16", -32768, 32767],
  5123: [2, "setUint16", 0, 65535],
  5126: [4, "setFloat32", -1, 1],
};
const normal = (v, t, n) => (n ? Math.max(v / types[t][3], types[t][2] < 0 ? -1 : 0) : v);
const error = (code) => (e) => e.code === code;
function fixture() {
  const json = {
    asset: { version: "2.0" },
    extensionsUsed: ["KHR_mesh_quantization"],
    extensionsRequired: ["KHR_mesh_quantization"],
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, translation: [7, 8, 9], scale: [0.5, 0.5, 0.5] }],
    meshes: [{ primitives: [{ attributes: {} }] }],
    buffers: [],
    bufferViews: [],
    accessors: [],
  };
  const buffers = [],
    p = json.meshes[0].primitives[0];
  function accessor(
    values,
    type = "VEC3",
    component = 5126,
    normalized = false,
    { stride, offset = 0 } = {},
  ) {
    const width = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[type],
      count = values.length / width,
      [size, set] = types[component];
    stride ??= Math.ceil((width * size) / 4) * 4;
    const data = new Uint8Array(offset + (count - 1) * stride + width * size),
      view = new DataView(data.buffer);
    values.forEach((v, i) =>
      view[set](offset + Math.floor(i / width) * stride + (i % width) * size, v, true),
    );
    const buffer = buffers.push(data) - 1;
    json.buffers.push({ byteLength: data.length });
    const bufferView =
      json.bufferViews.push({
        buffer,
        byteLength: data.length,
        ...(stride !== width * size ? { byteStride: stride } : {}),
      }) - 1;
    return (
      json.accessors.push({
        bufferView,
        byteOffset: offset,
        count,
        type,
        componentType: component,
        normalized,
      }) - 1
    );
  }
  p.attributes.POSITION = accessor([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  p.attributes.NORMAL = accessor([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return {
    json,
    buffers,
    p,
    accessor,
    decode: (options) => decodeGltfGeometry(json, buffers, options),
  };
}
for (const component of [5120, 5121, 5122, 5123])
  for (const normalized of [false, true]) {
    test(`quantized POSITION ${component} normalized=${normalized} retains numeric values and source transforms`, () => {
      const f = fixture(),
        values = [types[component][2], 1, types[component][3], 0, 2, 3, 4, 5, 6];
      f.p.attributes.POSITION = f.accessor(values, "VEC3", component, normalized, {
        offset: 4,
        stride: 16,
      });
      const a = f.json.accessors[f.p.attributes.POSITION];
      a.min = [100, 100, 100];
      a.max = [200, 200, 200];
      const before = structuredClone(f.json),
        bytes = f.buffers.map((b) => b.slice()),
        r = f.decode(),
        g = r.primitives[0].geometry;
      assert.deepEqual(
        [...g.positions],
        values.map((v) => normal(v, component, normalized)),
      );
      assert.ok(g.positions instanceof Float64Array);
      assert.deepEqual(f.json, before);
      f.buffers.forEach((b, i) => assert.deepEqual(b, bytes[i]));
      const definition = decodeGltfAnimation(f.json, f.buffers);
      assert.deepEqual(definition.nodes[0].translation, [7, 8, 9]);
      assert.deepEqual(definition.nodes[0].scale, [0.5, 0.5, 0.5]);
      assert.equal(r.primitives[0].node, 0);
      assert.deepEqual([...r.primitives[0].indices], [0, 1, 2]);
    });
    test(`quantized TEXCOORD ${component} normalized=${normalized} does not clamp or apply material transforms`, () => {
      const f = fixture(),
        values = [types[component][2], 1, types[component][3], 2, 3, 4];
      f.p.attributes.TEXCOORD_2 = f.accessor(values, "VEC2", component, normalized);
      assert.deepEqual(
        [...f.decode().primitives[0].attributes.TEXCOORD_2.values],
        values.map((v) => normal(v, component, normalized)),
      );
    });
  }
for (const component of [5120, 5122])
  for (const field of ["NORMAL", "TANGENT"]) {
    test(`signed-normalized ${field} ${component} preserves direction and handedness`, () => {
      const f = fixture(),
        lo = types[component][2],
        hi = types[component][3];
      const values =
        field === "NORMAL"
          ? [lo, 1, hi, 0, hi, 0, hi, 0, 0]
          : [hi, 0, 0, lo, 0, hi, 0, hi, 0, 0, hi, lo];
      f.p.attributes[field] = f.accessor(
        values,
        field === "NORMAL" ? "VEC3" : "VEC4",
        component,
        true,
      );
      const g = f.decode().primitives[0].geometry;
      assert.deepEqual(
        [...g[field === "NORMAL" ? "normals" : "tangents"]],
        values.map((v) => normal(v, component, true)),
      );
    });
    for (const normalized of [false, true])
      test(`morph ${field} ${component} normalized=${normalized} has explicit admission`, () => {
        const f = fixture();
        if (field === "TANGENT")
          f.p.attributes.TANGENT = f.accessor([1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1], "VEC4");
        const values = [-1, 0, 1, 1, 0, -1, 0, 1, 0];
        f.p.targets = [{ [field]: f.accessor(values, "VEC3", component, normalized) }];
        if (normalized)
          assert.deepEqual(
            [
              ...f.decode().primitives[0].geometry.morphTargets[0][
                field === "NORMAL" ? "normals" : "tangents"
              ],
            ],
            values.map((v) => normal(v, component, true)),
          );
        else assert.throws(() => f.decode(), error("GLTF_GEOMETRY_ATTRIBUTE"));
      });
  }
for (const component of [5120, 5121, 5122, 5123])
  for (const normalized of [false, true])
    test(`morph POSITION ${component} normalized=${normalized} preserves signed displacements only`, () => {
      const f = fixture(),
        signed = types[component][2] < 0,
        values = signed ? [-2, 1, 0, 1, -1, 0, 0, 0, 1] : [2, 1, 0, 1, 1, 0, 0, 0, 1];
      f.p.targets = [{ POSITION: f.accessor(values, "VEC3", component, normalized) }];
      if (signed)
        assert.deepEqual(
          [...f.decode().primitives[0].geometry.morphTargets[0].positions],
          values.map((v) => normal(v, component, normalized)),
        );
      else assert.throws(() => f.decode(), error("GLTF_GEOMETRY_ATTRIBUTE"));
    });
test("quantization requires its declaration; used-only never silently changes the core type contract", () => {
  for (const required of [[], undefined])
    for (const name of ["POSITION", "NORMAL", "TANGENT", "TEXCOORD_0"]) {
      const f = fixture();
      f.json.extensionsRequired = required;
      const width = name === "TANGENT" ? 4 : name.startsWith("TEXCOORD") ? 2 : 3,
        values = Array(width * 3).fill(1);
      f.p.attributes[name] = f.accessor(values, "VEC" + width, 5120, true);
      assert.throws(() => f.decode(), error("GLTF_GEOMETRY_ATTRIBUTE"));
    }
});
test("core unsigned normalized UVs need no extension and integer colors/weights are not broadened", () => {
  const f = fixture();
  delete f.json.extensionsRequired;
  delete f.json.extensionsUsed;
  f.p.attributes.TEXCOORD_0 = f.accessor([0, 255, 128, 0, 255, 1], "VEC2", 5121, true);
  assert.equal(f.decode().primitives[0].attributes.TEXCOORD_0.values[2], 128 / 255);
  f.json.extensionsRequired = ["KHR_mesh_quantization"];
  f.p.attributes.COLOR_0 = f.accessor(Array(9).fill(1), "VEC3", 5120, true);
  assert.throws(() => f.decode(), error("GLTF_GEOMETRY_ATTRIBUTE"));
});
test("invalid directions, tangent handedness, counts and byte alignment fail before output", () => {
  for (const mutate of [
    (f) => {
      f.p.attributes.NORMAL = f.accessor(Array(9).fill(1), "VEC3", 5121, true);
    },
    (f) => {
      f.p.attributes.NORMAL = f.accessor(Array(9).fill(1), "VEC3", 5122, false);
    },
    (f) => {
      f.p.attributes.TANGENT = f.accessor(Array(12).fill(1), "VEC4", 5120, true);
    },
    (f) => {
      f.p.attributes.POSITION = f.accessor(Array(9).fill(1), "VEC3", 5120, false, { stride: 3 });
    },
    (f) => {
      f.p.attributes.POSITION = f.accessor(Array(9).fill(1), "VEC3", 5120, false, { offset: 1 });
    },
    (f) => {
      f.p.attributes.NORMAL = f.accessor([0, 0, 127], "VEC3", 5120, true);
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => f.decode());
  }
});
test("quantized sparse overlay uses declared normalization without mutating compressed values", () => {
  const f = fixture(),
    a = f.accessor([0, 0, 0, 0, 0, 0, 0, 0, 0], "VEC3", 5122, true);
  f.p.attributes.POSITION = a;
  const indices = f.accessor([1], "SCALAR", 5121),
    values = f.accessor([-32768, 16384, 32767], "VEC3", 5122);
  for (const i of [indices, values])
    delete f.json.bufferViews[f.json.accessors[i].bufferView].byteStride;
  f.json.accessors[a].sparse = {
    count: 1,
    indices: { bufferView: f.json.accessors[indices].bufferView, componentType: 5121 },
    values: { bufferView: f.json.accessors[values].bufferView },
  };
  assert.deepEqual(
    [...f.decode().primitives[0].geometry.positions],
    [0, 0, 0, -1, 16384 / 32767, 1, 0, 0, 0],
  );
});
test("quantized skin geometry preserves inverse-bind dequantization and all influences", () => {
  const f = fixture();
  f.json.nodes.push({});
  f.json.scenes[0].nodes.push(1);
  f.json.nodes[0].skin = 0;
  const ibm = [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 3, 4, 5, 1];
  f.json.skins = [{ joints: [1], inverseBindMatrices: f.accessor(ibm, "MAT4") }];
  f.p.attributes.POSITION = f.accessor([0, 0, 0, 100, 0, 0, 0, 100, 0], "VEC3", 5123);
  f.p.attributes.JOINTS_0 = f.accessor(Array(12).fill(0), "VEC4", 5121);
  f.p.attributes.WEIGHTS_0 = f.accessor(
    [255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0],
    "VEC4",
    5121,
    true,
  );
  const g = f.decode().primitives[0].geometry;
  assert.equal(g.positions[3], 100);
  assert.equal(g.influences, 4);
  assert.deepEqual([...g.weights], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  assert.deepEqual(decodeGltfAnimation(f.json, f.buffers).skins[0].inverseBindMatrices, ibm);
});
test("quantized morphs keep dynamic flat-normal expansion, topology and per-instance ownership", () => {
  const f = fixture();
  delete f.p.attributes.NORMAL;
  f.p.attributes.POSITION = f.accessor([0, 0, 0, 10, 0, 0, 0, 10, 0], "VEC3", 5122);
  f.p.targets = [{ POSITION: f.accessor([0, 0, 0, 0, 0, 0, 0, 0, 10], "VEC3", 5120) }];
  f.json.nodes.push({ mesh: 0 });
  f.json.scenes[0].nodes.push(1);
  f.p.mode = 6;
  const result = f.decode(),
    [a, b] = result.primitives;
  assert.equal(a.geometry.flatNormals, true);
  assert.equal(a.indices, null);
  assert.deepEqual([...a.geometry.positions], [10, 0, 0, 0, 10, 0, 0, 0, 0]);
  assert.deepEqual([...a.geometry.morphTargets[0].positions], [0, 0, 0, 0, 0, 10, 0, 0, 0]);
  a.geometry.positions[0] = 77;
  assert.equal(b.geometry.positions[0], 10);
  assert.equal(result.diagnostics.filter((d) => d.reason === "DYNAMIC_FLAT_NORMALS").length, 2);
});
test("output and accessor limits still charge decoded components, not the smaller encoded byte count", () => {
  const f = fixture();
  f.p.attributes.POSITION = f.accessor([0, 0, 0, 127, 0, 0, 0, 127, 0], "VEC3", 5120, true);
  assert.throws(() => f.decode({ maxComponents: 17 }), error("GLTF_ANIMATION_LIMIT"));
  assert.throws(() => f.decode({ maxComponents: 20 }), error("GLTF_GEOMETRY_LIMIT"));
  assert.equal(f.decode({ maxComponents: 21 }).outputComponents, 21);
});
