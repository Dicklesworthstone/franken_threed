import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationDeformer } from "./animation_deformer.mjs";
import { decodeGltfGeometry } from "./animation_geometry.mjs";
import { createGltfAccessorReader, decodeGltfAnimation } from "./animation_gltf.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

const code = (code) => (error) => error.code === code;
const close = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) =>
    assert.ok(Math.abs(v - expected[i]) < 1e-6, `${i}: ${v} != ${expected[i]}`),
  );
};

function fixture() {
  const model = {
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: {} }] }],
      buffers: [],
      bufferViews: [],
      accessors: [],
    },
    buffers = [];
  function accessor(values, type = "VEC3", componentType = 5126, options = {}) {
    const width = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[type],
      count = values.length / width;
    const [size, setter] = {
      5120: [1, "setInt8"],
      5121: [1, "setUint8"],
      5122: [2, "setInt16"],
      5123: [2, "setUint16"],
      5125: [4, "setUint32"],
      5126: [4, "setFloat32"],
    }[componentType];
    const stride = options.stride ?? width * size,
      offset = options.offset ?? 0,
      bytes = new ArrayBuffer(offset + (count - 1) * stride + width * size),
      data = new DataView(bytes);
    for (let v = 0; v < count; v++)
      for (let c = 0; c < width; c++)
        data[setter](offset + v * stride + c * size, values[v * width + c], true);
    const buffer = buffers.push(bytes) - 1,
      view =
        model.bufferViews.push({
          buffer,
          byteLength: bytes.byteLength,
          ...(options.stride ? { byteStride: stride } : {}),
        }) - 1;
    model.buffers.push({ byteLength: bytes.byteLength });
    return (
      model.accessors.push({
        bufferView: view,
        byteOffset: offset,
        componentType,
        count,
        type,
        ...(options.normalized ? { normalized: true } : {}),
        ...(options.min ? { min: options.min, max: options.max } : {}),
      }) - 1
    );
  }
  const primitive = model.meshes[0].primitives[0];
  primitive.attributes.POSITION = accessor([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  model.accessors[0].min = [0, 0, 0];
  model.accessors[0].max = [1, 1, 0];
  primitive.attributes.NORMAL = accessor([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return {
    model,
    buffers,
    primitive,
    accessor,
    decode: (options) => decodeGltfGeometry(model, buffers, options),
  };
}

test("default scene decodes real geometry without mutating its source buffers", () => {
  const f = fixture(),
    before = f.buffers.map((b) => new Uint8Array(b).slice()),
    r = f.decode(),
    p = r.primitives[0];
  assert.equal(r.format, "f3d-gltf-geometry-v1");
  assert.equal(r.scene, 0);
  assert.deepEqual([p.node, p.mesh, p.primitive, p.material], [0, 0, 0, null]);
  close([...p.geometry.positions], [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepEqual([...p.indices], [0, 1, 2]);
  assert.equal(p.geometry.positions, p.attributes.POSITION.values);
  p.geometry.positions.fill(99);
  f.buffers.forEach((b, i) => assert.deepEqual(new Uint8Array(b), before[i]));
});

test("mesh instances and primitives retain separate node identities and writable geometry", () => {
  const f = fixture();
  f.model.nodes = [
    { children: [1, 2] },
    { mesh: 0, translation: [1, 0, 0] },
    { mesh: 0, translation: [2, 0, 0] },
  ];
  f.model.meshes[0].primitives.push(structuredClone(f.primitive));
  const r = f.decode();
  assert.deepEqual(
    r.primitives.map((p) => [p.node, p.mesh, p.primitive]),
    [
      [1, 0, 0],
      [1, 0, 1],
      [2, 0, 0],
      [2, 0, 1],
    ],
  );
  const first = r.primitives[0].geometry.positions;
  r.primitives[1].geometry.positions[0] = 10;
  assert.equal(first[0], 0);
  assert.equal(r.accessorComponents, 18);
  assert.equal(r.outputComponents, 84);
});

test("selected-scene reachability avoids decoding unused mesh buffers", () => {
  const f = fixture();
  f.model.nodes.push({ mesh: 1 });
  f.model.scenes.push({ nodes: [1] });
  f.model.meshes.push({ primitives: [{ attributes: { POSITION: 999 } }] });
  let loads = 0;
  const r = decodeGltfGeometry(f.model, (i) => {
    loads++;
    return f.buffers[i];
  });
  assert.equal(loads, 2);
  assert.equal(r.primitives.length, 1);
  assert.throws(() => f.decode({ scene: 1 }), code("GLTF_ANIMATION_INDEX"));
});

for (const [mode, expected] of [
  [4, [0, 1, 2, 2, 1, 3]],
  [5, [0, 1, 2, 1, 3, 2]],
  [6, [1, 2, 0, 2, 3, 0]],
])
  test(`topology ${mode} preserves winding in triangle-list conversion`, () => {
    const f = fixture();
    f.primitive.attributes.POSITION = f.accessor([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    f.primitive.attributes.NORMAL = f.accessor(Array.from({ length: 4 }, () => [0, 0, 1]).flat());
    f.primitive.mode = mode;
    f.primitive.indices = f.accessor(
      mode === 4 ? [0, 1, 2, 2, 1, 3] : [0, 1, 2, 3],
      "SCALAR",
      5123,
    );
    assert.deepEqual([...f.decode().primitives[0].indices], expected);
  });

test("unindexed strips and fans use all vertices, retaining degenerate triangles", () => {
  const f = fixture();
  f.primitive.mode = 5;
  f.primitive.indices = f.accessor([0, 0, 1, 2], "SCALAR", 5121);
  assert.deepEqual([...f.decode().primitives[0].indices], [0, 0, 1, 0, 2, 1]);
  delete f.primitive.indices;
  f.primitive.mode = 6;
  assert.deepEqual([...f.decode().primitives[0].indices], [1, 2, 0]);
});

test("interleaved data and normalized UV/color attributes use the shared accessor reader", () => {
  const f = fixture();
  f.primitive.attributes.POSITION = f.accessor([0, 0, 0, 1, 0, 0, 0, 1, 0], "VEC3", 5126, {
    stride: 20,
    offset: 4,
  });
  f.primitive.attributes.TEXCOORD_1 = f.accessor([0, 65535, 32768, 0, 65535, 65535], "VEC2", 5123, {
    normalized: true,
  });
  f.primitive.attributes.COLOR_0 = f.accessor([255, 0, 128, 0, 255, 0, 0, 0, 255], "VEC3", 5121, {
    normalized: true,
    stride: 4,
  });
  const p = f.decode().primitives[0];
  close([...p.attributes.TEXCOORD_1.values], [0, 1, 32768 / 65535, 0, 1, 1]);
  close([...p.attributes.COLOR_0.values], [1, 0, 128 / 255, 0, 1, 0, 0, 0, 1]);
  assert.equal(p.attributes.COLOR_0.width, 3);
});

test("sparse positions overlay zero initialization and reject duplicate sparse indices", () => {
  const f = fixture(),
    sparseIndices = f.accessor([1, 2], "SCALAR", 5121),
    sparseValues = f.accessor([1, 0, 0, 0, 1, 0]);
  f.model.accessors[0] = {
    count: 3,
    type: "VEC3",
    componentType: 5126,
    min: [0, 0, 0],
    max: [1, 1, 0],
    sparse: {
      count: 2,
      indices: { bufferView: f.model.accessors[sparseIndices].bufferView, componentType: 5121 },
      values: { bufferView: f.model.accessors[sparseValues].bufferView },
    },
  };
  close([...f.decode().primitives[0].geometry.positions], [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  new Uint8Array(
    f.buffers[f.model.bufferViews[f.model.accessors[sparseIndices].bufferView].buffer],
  ).set([1, 1]);
  assert.throws(() => f.decode(), code("GLTF_ANIMATION_SPARSE"));
});

test("flat normals expand all attributes with source triangle orientation and ignore base tangents", () => {
  const f = fixture();
  delete f.primitive.attributes.NORMAL;
  f.primitive.indices = f.accessor([0, 1, 2, 0, 2, 1], "SCALAR", 5121);
  f.primitive.attributes.TANGENT = f.accessor([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], "VEC4");
  f.primitive.attributes.TEXCOORD_0 = f.accessor([0, 0, 1, 0, 0, 1], "VEC2");
  const r = f.decode(),
    p = r.primitives[0];
  assert.equal(p.indices, null);
  assert.equal(p.geometry.positions.length, 18);
  close([...p.geometry.normals], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1]);
  close([...p.attributes.TEXCOORD_0.values], [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]);
  assert.equal(p.geometry.tangents, undefined);
  assert.deepEqual(
    r.diagnostics.map((d) => d.reason),
    ["IGNORED_TANGENTS_WITHOUT_NORMALS", "GENERATED_FLAT_NORMALS"],
  );
});

test("morph positions/normals/tangent deltas preserve target order and pose weight association", () => {
  const f = fixture();
  f.primitive.attributes.TANGENT = f.accessor([1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1], "VEC4");
  const position = f.accessor([0, 0, 2, 0, 0, 2, 0, 0, 2]),
    normal = f.accessor([0, 1, 0, 0, 1, 0, 0, 1, 0]),
    tangent = f.accessor([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  f.primitive.targets = [{ POSITION: position, NORMAL: normal, TANGENT: tangent }, {}];
  f.model.meshes[0].weights = [0.25, 0.75];
  const p = f.decode().primitives[0],
    pose = createAnimationPlayer(decodeGltfAnimation(f.model, f.buffers));
  assert.deepEqual([...pose.morphWeights], [0.25, 0.75]);
  assert.equal(p.geometry.morphTargets.length, 2);
  assert.equal(p.geometry.morphTargets[0].tangents.length, 9);
  assert.equal(p.geometry.tangents[3], -1);
  assert.deepEqual(p.geometry.morphTargets[1], {});
  pose.dispose();
});

function skinFixture() {
  const f = fixture();
  f.model.nodes.push({ translation: [2, 0, 0] }, { translation: [0, 4, 0] });
  f.model.scenes[0].nodes = [0, 1, 2];
  f.model.skins = [{ joints: [1, 2] }];
  f.model.nodes[0].skin = 0;
  f.primitive.attributes.JOINTS_0 = f.accessor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "VEC4", 5121);
  f.primitive.attributes.WEIGHTS_0 = f.accessor(
    [128, 0, 0, 0, 128, 0, 0, 0, 128, 0, 0, 0],
    "VEC4",
    5121,
    { normalized: true },
  );
  f.primitive.attributes.JOINTS_1 = f.accessor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "VEC4", 5121);
  f.primitive.attributes.WEIGHTS_1 = f.accessor(
    [127, 0, 0, 0, 127, 0, 0, 0, 127, 0, 0, 0],
    "VEC4",
    5121,
    { normalized: true },
  );
  return f;
}

test("eight skin influences combine vertex-major without dropping secondary sets", () => {
  const f = skinFixture(),
    p = f.decode().primitives[0];
  assert.equal(p.geometry.influences, 8);
  assert.deepEqual(
    [...p.geometry.joints],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  );
  for (let v = 0; v < 3; v++)
    close(
      [...p.geometry.weights.slice(v * 8, v * 8 + 8)],
      [128 / 255, 0, 0, 0, 127 / 255, 0, 0, 0],
    );
});

test("mesh shared by skinned and rigid nodes keeps skin data only on skinned deformer", () => {
  const f = skinFixture();
  f.model.nodes.push({ mesh: 0 });
  f.model.scenes[0].nodes.push(3);
  const r = f.decode();
  assert.equal(r.primitives[0].geometry.influences, 8);
  assert.equal(r.primitives[1].geometry.influences, undefined);
  assert.ok(r.primitives[1].attributes.JOINTS_0);
});

for (const change of [
  (f) => {
    delete f.primitive.attributes.WEIGHTS_1;
  },
  (f) => {
    f.primitive.attributes.JOINTS_2 = f.primitive.attributes.JOINTS_1;
    delete f.primitive.attributes.JOINTS_1;
  },
  (f) => {
    f.primitive.attributes.JOINTS_1 = f.primitive.attributes.JOINTS_0;
  },
  (f) => {
    f.model.skins[0].joints = [1];
  },
  (f) => {
    delete f.primitive.attributes.JOINTS_1;
    delete f.primitive.attributes.WEIGHTS_1;
  },
])
  test("invalid or incomplete skin sets fail rather than truncate or normalize", () => {
    const f = skinFixture();
    change(f);
    assert.throws(() => f.decode(), code("GLTF_GEOMETRY_SKIN"));
  });

test("accessor and emitted budgets include aliases, indices and flat-normal expansion", () => {
  const f = fixture();
  assert.throws(() => f.decode({ maxComponents: 17 }), code("GLTF_ANIMATION_LIMIT"));
  assert.throws(() => f.decode({ maxComponents: 20 }), code("GLTF_GEOMETRY_LIMIT"));
  assert.equal(f.decode({ maxComponents: 21 }).outputComponents, 21);
  f.model.nodes.push({ mesh: 0 });
  f.model.scenes[0].nodes.push(1);
  assert.throws(() => f.decode({ maxComponents: 41 }), code("GLTF_GEOMETRY_LIMIT"));
  assert.throws(() => f.decode({ maxPrimitives: 1 }), code("GLTF_GEOMETRY_LIMIT"));
  assert.equal(f.decode({ maxComponents: 42 }).outputComponents, 42);
});

for (const mutate of [
  (f) => {
    f.model.nodes[0].children = [0];
  },
  (f) => {
    f.model.nodes.push({ children: [2] }, { children: [1] });
  },
  (f) => {
    f.model.nodes.push({});
    f.model.nodes[0].children = [1, 1];
  },
  (f) => {
    f.model.scenes[0].nodes = [0, 0];
  },
  (f) => {
    f.model.nodes.push({});
    f.model.nodes[0].children = [1];
    f.model.scenes[0].nodes = [1];
  },
])
  test("malformed scene forests fail before buffer access", () => {
    const f = fixture();
    mutate(f);
    assert.throws(
      () =>
        decodeGltfGeometry(f.model, () => {
          assert.fail("no buffer reads expected");
        }),
      code("GLTF_GEOMETRY_HIERARCHY"),
    );
  });

test("deep valid forests use iterative traversal rather than the call stack", () => {
  const f = fixture();
  f.model.nodes = Array.from({ length: 12000 }, (_, i) =>
    i === 11999 ? { mesh: 0 } : { children: [i + 1] },
  );
  assert.equal(f.decode().primitives[0].node, 11999);
});

test("bad topology, indices, mixed attribute counts and compressed primitives fail explicitly", () => {
  for (const [mutate, expected] of [
    [
      (f) => {
        f.primitive.mode = 1;
      },
      "GLTF_GEOMETRY_TOPOLOGY",
    ],
    [
      (f) => {
        f.primitive.indices = f.accessor([0, 1], "SCALAR", 5121);
      },
      "GLTF_GEOMETRY_TOPOLOGY",
    ],
    [
      (f) => {
        f.primitive.indices = f.accessor([0, 1, 3], "SCALAR", 5121);
      },
      "GLTF_GEOMETRY_INDEX",
    ],
    [
      (f) => {
        f.primitive.attributes.NORMAL = f.accessor([0, 0, 1]);
      },
      "GLTF_GEOMETRY_ATTRIBUTE",
    ],
    [
      (f) => {
        f.primitive.extensions = { KHR_draco_mesh_compression: {} };
      },
      "GLTF_GEOMETRY_EXTENSION",
    ],
    [
      (f) => {
        f.primitive.targets = [{ COLOR_0: 0 }];
      },
      "GLTF_GEOMETRY_MORPH",
    ],
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => f.decode(), code(expected));
  }
});

test("source custom attributes are retained, not interpreted as rendering features", () => {
  const f = fixture();
  f.primitive.attributes._TEMPERATURE = f.accessor([1, 2, 3], "SCALAR");
  assert.deepEqual([...f.decode().primitives[0].attributes._TEMPERATURE.values], [1, 2, 3]);
  f.primitive.attributes.unrecognized = 0;
  assert.throws(() => f.decode(), code("GLTF_GEOMETRY_ATTRIBUTE"));
});

test("deforming meshes without normals do not receive incorrectly static flat normals", () => {
  const f = skinFixture();
  delete f.primitive.attributes.NORMAL;
  assert.equal(f.decode().primitives[0].geometry.flatNormals, true);
  const g = fixture();
  delete g.primitive.attributes.NORMAL;
  g.primitive.targets = [{ POSITION: g.accessor(Array(9).fill(0)) }];
  assert.equal(g.decode().primitives[0].geometry.flatNormals, true);
});

test("shared accessor cache has one bounded decode and preserves little-endian float values", () => {
  const f = fixture();
  let reads = 0;
  const reader = createGltfAccessorReader(
    f.model,
    (i) => {
      reads++;
      return f.buffers[i];
    },
    { maxComponents: 9 },
  );
  const a = reader.read(0);
  assert.equal(reader.read(0), a);
  assert.equal(reads, 1);
  assert.equal(reader.components, 9);
  assert.throws(() => reader.read(1), code("GLTF_ANIMATION_LIMIT"));
});

function animatedFlatFixture({ skin = true, morph = true, mode = 4, indexed = true } = {}) {
  const f = fixture();
  delete f.primitive.attributes.NORMAL;
  f.primitive.mode = mode;
  if (indexed)
    f.primitive.indices = f.accessor(
      [0, 1, 2, 0, 2, 1].slice(0, mode === 4 ? 6 : 4),
      "SCALAR",
      5123,
    );
  f.primitive.attributes.TEXCOORD_1 = f.accessor([0, 0, 1, 0, 0, 1], "VEC2");
  const times = f.accessor([0, 1], "SCALAR", 5126, { min: [0], max: [1] }),
    samplers = [],
    channels = [];
  function channel(node, path, values, type) {
    const output = f.accessor(values, type);
    channels.push({ sampler: samplers.length, target: { node, path } });
    samplers.push({ input: times, output });
  }
  if (morph) {
    f.primitive.targets = [{ POSITION: f.accessor([0, 0, 0, 0, 0, 0, 0, 0, 1]) }];
    f.model.meshes[0].weights = [0];
    channel(0, "weights", [0, 1], "SCALAR");
  }
  if (skin) {
    f.model.nodes.push({}, {});
    f.model.scenes[0].nodes.push(1, 2);
    f.model.skins = [{ joints: [1, 2] }];
    f.model.nodes[0].skin = 0;
    f.primitive.attributes.JOINTS_0 = f.accessor(
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      "VEC4",
      5121,
    );
    f.primitive.attributes.WEIGHTS_0 = f.accessor(
      [255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0],
      "VEC4",
      5121,
      { normalized: true },
    );
    channel(2, "translation", [0, 0, 0, 0, 0, 2], "VEC3");
  }
  f.model.animations = [{ samplers, channels }];
  return f;
}

for (const skin of [false, true])
  for (const morph of [false, true])
    test(`binary glTF flat geometry runs with skin=${skin}, morph=${morph}`, () => {
      const f = animatedFlatFixture({ skin, morph }),
        before = f.buffers.map((b) => new Uint8Array(b).slice());
      const decoded = f.decode(),
        primitive = decoded.primitives[0],
        pose = createAnimationPlayer(decodeGltfAnimation(f.model, f.buffers));
      const deformer = createAnimationDeformer(pose, primitive.geometry);
      assert.equal(primitive.indices, null);
      assert.equal(primitive.geometry.flatNormals, skin || morph ? true : undefined);
      assert.deepEqual(
        [primitive.node, primitive.mesh, primitive.primitive, primitive.material],
        [0, 0, 0, null],
      );
      assert.equal(
        decoded.diagnostics.some((d) => d.reason === "DYNAMIC_FLAT_NORMALS"),
        skin || morph,
      );
      const normals = deformer.normals;
      for (const t of [0, 0.5, 1]) {
        pose.sample(t);
        deformer.update();
        assert.equal(deformer.normals, normals);
        const x = skin ? -2 * t : 0,
          y = morph ? -t : 0,
          length = Math.hypot(x, y, 1),
          n = [x / length, y / length, 1 / length];
        for (let v = 0; v < 6; v++)
          close(
            [...normals.slice(v * 3, v * 3 + 3)],
            n.map((a) => (v < 3 ? a : -a)),
          );
      }
      close([...primitive.attributes.TEXCOORD_1.values], [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]);
      f.buffers.forEach((buffer, i) => assert.deepEqual(new Uint8Array(buffer), before[i]));
      deformer.dispose();
      pose.dispose();
    });

for (const mode of [4, 5, 6])
  for (const indexed of [false, true])
    test(`dynamic flat topology ${mode}, indexed=${indexed}, preserves face winding and pose`, () => {
      const f = animatedFlatFixture({ mode, indexed }),
        p = f.decode().primitives[0];
      const pose = createAnimationPlayer(decodeGltfAnimation(f.model, f.buffers)),
        d = createAnimationDeformer(pose, p.geometry);
      pose.sample(1);
      d.update();
      assert.equal(p.indices, null);
      assert.equal(d.positions.length % 9, 0);
      for (let i = 0; i < d.positions.length; i += 9) {
        const a = [0, 1, 2].map((k) => d.positions[i + 3 + k] - d.positions[i + k]);
        const b = [0, 1, 2].map((k) => d.positions[i + 6 + k] - d.positions[i + k]);
        const n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
          length = Math.hypot(...n);
        for (let v = 0; v < 3; v++)
          close(
            [...d.normals.slice(i + v * 3, i + v * 3 + 3)],
            n.map((x) => (length ? x / length : 0)),
          );
      }
      d.dispose();
      pose.dispose();
    });

test("dynamic flat expansion is charged fully for streams, weights and repeated instances", () => {
  const f = animatedFlatFixture();
  f.model.nodes.push({ mesh: 0, skin: 0 });
  f.model.scenes[0].nodes.push(3);
  const result = f.decode(),
    required = result.outputComponents;
  assert.equal(result.primitives.length, 2);
  assert.notEqual(result.primitives[0].geometry.positions, result.primitives[1].geometry.positions);
  assert.equal(f.decode({ maxComponents: required }).outputComponents, required);
  assert.throws(() => f.decode({ maxComponents: required - 1 }), code("GLTF_GEOMETRY_LIMIT"));
});

test("authored normals still keep indexed geometry and the original morph-normal contract", () => {
  const f = animatedFlatFixture();
  f.primitive.attributes.NORMAL = f.accessor([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  f.primitive.targets[0].NORMAL = f.accessor([0, 1, 0, 0, 1, 0, 0, 1, 0]);
  const p = f.decode().primitives[0],
    pose = createAnimationPlayer(decodeGltfAnimation(f.model, f.buffers)),
    d = createAnimationDeformer(pose, p.geometry);
  assert.equal(p.geometry.flatNormals, undefined);
  assert.deepEqual([...p.indices], [0, 1, 2, 0, 2, 1]);
  pose.sample(1);
  d.update();
  close([...d.normals], [0, 1, 1, 0, 1, 1, 0, 1, 1]);
  d.dispose();
  pose.dispose();
});

test("source tangents and tangent deltas are ignored together when flat normals are required", () => {
  const f = animatedFlatFixture();
  f.primitive.attributes.TANGENT = f.accessor([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], "VEC4");
  f.primitive.targets[0].TANGENT = f.accessor(Array(9).fill(0));
  const p = f.decode().primitives[0];
  assert.equal(p.geometry.tangents, undefined);
  assert.equal(p.geometry.morphTargets[0].tangents, undefined);
  assert.equal(p.geometry.flatNormals, true);
});
